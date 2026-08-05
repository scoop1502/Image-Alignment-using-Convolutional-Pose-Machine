import os
import cv2
import torch
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm



# Import your model architecture

from cpm_model_v2 import CPM


def gaussian(x, y, H, W, sigma=5):
    """
    Create a center map by convoluting a 2D gaussian kernel
    """
    channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))
    return channel


def heatmap2coord(heatmap, topk=5):

    N, C, H, W = heatmap.shape

    score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)

    coord = torch.cat([index%W, index//W], dim=2)

    coord = (coord*torch.nn.functional.softmax(score, dim=-1)).sum(-1) * 8

    return coord

    

def draw_lmk(heatmap, lmk=None, img=None):

    if lmk is None:

        lmk = heatmap2coord(heatmap[:, 1:, ], topk = 5).numpy()[0]

    

    if img is None:

        tmp = np.zeros((224, 224, 3), dtype=np.uint8)

    else:

        tmp = img.copy()

    for i, xy in enumerate(lmk):

        x, y = int(xy[0]), int(xy[1])

        cv2.circle(tmp, (x, y), 3, (255, 0, 0), -1)

        tmp = cv2.putText(tmp, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)



    return tmp



# --- Configuration ---

INFERENCE_DIR = "data/images/inference"

OUTPUT_DIR = "output/inference"

CHECKPOINT_PATH = "epoch-latest.pth"

TARGET_SIZE = 224



os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.ToTensor()



# --- Model Initialization ---

model = CPM(5)

model = model.cuda()

model.eval()



# Load weights

if os.path.exists(CHECKPOINT_PATH):

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")["state_dict"]

    state_dict = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}

    model.load_state_dict(state_dict)

    print("Model weights loaded.")

else:

    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")



# --- Inference Loop ---

image_files = [f for f in os.listdir(INFERENCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- Generate Static Center Map ---
centermap_np = np.zeros((TARGET_SIZE, TARGET_SIZE, 1), dtype=np.float32)
centermap_np[:, :, 0] = gaussian(112, 112, TARGET_SIZE, TARGET_SIZE)
centermap_np = centermap_np.transpose(2, 0, 1) # Shape becomes (1, 224, 224)
center_map_tensor = torch.from_numpy(centermap_np).unsqueeze(0).cuda() # Add batch dimension -> (1, 1, 224, 224)

for filename in tqdm(image_files, desc="Processing Images"):

    image_path = os.path.join(INFERENCE_DIR, filename)

    

    # 1. Load the raw image[cite: 4]

    raw_image = Image.open(image_path).convert("RGB")

    w, h = raw_image.size

    

    # 2. Calculate scaling factor to fit within target_size[cite: 4]

    scale_x = TARGET_SIZE / w

    scale_y = TARGET_SIZE / h

    scale = min(scale_x, scale_y)

    

    # Resize image maintaining aspect ratio[cite: 4]

    image_np = np.asarray(raw_image)

    image_resized = cv2.resize(image_np, None, fx=scale, fy=scale)

    

    # 3. Pad image into a 224x224 array (zero-padding on bottom/right)[cite: 4]

    resized_h, resized_w, _ = image_resized.shape
    box_image = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    box_image[:resized_h, :resized_w, :] = image_resized

    

    # 4. Convert back to PIL and apply ToTensor transform[cite: 4]

    image_pil = Image.fromarray(box_image)

    input_tensor = transform(image_pil).unsqueeze(0).cuda()

    

    # 5. Model Prediction

    with torch.no_grad():

        heatmaps = model(input_tensor, center_map_tensor)

        # Check if the output is a list or a tuple to extract the final stage
        if isinstance(heatmaps, (list, tuple)):
            final_heatmap = heatmaps[-1]
        else:
            final_heatmap = heatmaps
            
        # In some architectures, the final stage itself might be returned as a tuple
        if isinstance(final_heatmap, (list, tuple)):
            final_heatmap = final_heatmap[0]

            

    # 6. Convert predictions to coordinates

    pred_coords = heatmap2coord(final_heatmap.cpu())

    

    # 7. Visualization

    # Convert RGB box_image to BGR for OpenCV saving

    img_bgr = cv2.cvtColor(box_image, cv2.COLOR_RGB2BGR)

    visualized_img = draw_lmk(final_heatmap.cpu(), lmk=pred_coords.numpy()[0], img=img_bgr)

    

    # Save output

    output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")

    cv2.imwrite(output_path, visualized_img)