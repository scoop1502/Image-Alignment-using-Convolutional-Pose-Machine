"""
Inference for CPM keypoint model on unlabelled / unannotated images.

The model was trained on *crops* of the insect (see insect_dataset.py: the image is
cropped to the annotated bounding box, then resized so the long side is 224 and
zero-padded bottom/right).  So at inference time we must supply a box too.
Without annotations the box comes from one of:

    --box-mode auto    : segment the insect automatically (default)
    --box-mode full    : use the whole image (only correct if images are already crops)
    --box-mode file    : read boxes from a JSON file {"filename.jpg": [x1, y1, x2, y2], ...}

The input directory is walked recursively and any subdirectory structure -- e.g. one
folder per class -- is mirrored in both output trees, so labels carried by the folder
layout survive the export.  The CSV names images by their path relative to --input.

Outputs, per image (<sub> being the image's subdirectory under --input, if any):
    <out>/<sub>/pred_<name>.jpg   visualisation on the ORIGINAL image
    <out>/keypoints.csv           keypoint coords in ORIGINAL image pixels + confidence

Optional exports for downstream ViT training (see docs/inference_guide.pdf):
    --save-crops DIR    pose-normalised specimen crops -- cropped AND rotated/scaled
                        (MAE pre-training input).  Rotation comes from two predicted
                        keypoints; zoom comes from the detected box, so every specimen
                        fills the same fraction of the frame no matter how close the
                        original shot was.  See --align-scale-mode.
    --list-file PATH    newline-separated image paths for ImageDatasetListV2

Usage:
    python infer_unlabeled.py --input data/images/inference --output output/inference
"""

import os
import csv
import json
import math
import inspect
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from cpm_model_v2 import CPM

TARGET_SIZE = 224
STRIDE = 8  # 224 / 28, the heatmap downsampling factor
N_POINTS = 5

# insect_dataset.py builds heatmaps from lmks[:, :, [3,4,5,6,7]], i.e. the 8-point
# annotation's keypoints 4..8 (1-indexed).  Channel 0 of the model output is background.
KP_NAMES = ["kp4", "kp5", "kp6", "kp7", "kp8"]


def gaussian(x, y, H, W, sigma=5):
    """2D gaussian kernel centred at (x, y) -- same helper as insect_dataset.py."""
    channel = [math.exp(-((c - x) ** 2 + (r - y) ** 2) / (2 * sigma ** 2))
               for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    # Positional, not newshape=: NumPy 2.1 renamed that keyword to `shape`.
    return np.reshape(channel, (H, W))


def heatmap2coord(heatmap, topk=5):
    """Soft-argmax over the topk peaks. Returns coords in 224-space."""
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N, C, 1, -1).topk(topk, dim=-1)
    coord = torch.cat([index % W, torch.div(index, W, rounding_mode="floor")], dim=2)
    coord = (coord * torch.nn.functional.softmax(score, dim=-1)).sum(-1) * STRIDE
    return coord


def detect_box(image_np, margin=0.15, min_area_frac=1e-4):
    """
    Locate the insect in a raw frame.

    These are microscope shots: the specimen is coloured (tan/brown) while the
    background -- paper, grid lines, dust -- is essentially achromatic.  So we
    threshold the HSV saturation channel and keep the largest blob.

    Returns ((x1, y1, x2, y2), side) -- the box clipped to the frame, plus the
    *unclipped* square side it was built from.  Alignment needs the latter: a
    specimen wider than the frame is tall (the lateral beetle shots) has its box
    clipped vertically, so the clipped box understates the animal's true extent
    and would make it come out too small.  Returns None if nothing convincing
    was found.
    """
    h, w = image_np.shape[:2]
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    sat = cv2.GaussianBlur(sat, (9, 9), 0)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area_frac * h * w:
        return None

    bx, by, bw, bh = cv2.boundingRect(biggest)

    # Legs and antennae are thin and pale, so the saturation blob under-covers the
    # animal.  Pad outwards, and square the box up -- training crops were whole
    # animals, not tight body-only boxes.
    side = max(bw, bh) * (1.0 + 2 * margin)
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    x1 = max(0.0, cx - side / 2.0)
    y1 = max(0.0, cy - side / 2.0)
    x2 = min(float(w), cx + side / 2.0)
    y2 = min(float(h), cy + side / 2.0)
    return (x1, y1, x2, y2), side


def preprocess(pil_image, box):
    """
    Reproduce insect_dataset.__getitem__ preprocessing exactly:
    crop to box -> resize by min(224/w, 224/h) -> paste top-left into a 224x224 canvas.

    Returns (canvas_rgb_uint8, scale, x1, y1) -- the last three let us map
    predictions back to original-image pixels.
    """
    w, h = pil_image.size
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)

    crop = pil_image.crop([x1, y1, x2, y2])
    cw, ch = crop.size
    scale = min(TARGET_SIZE / cw, TARGET_SIZE / ch)

    resized = cv2.resize(np.asarray(crop), None, fx=scale, fy=scale)
    rh, rw, _ = resized.shape

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    canvas[:rh, :rw, :] = resized
    return canvas, scale, x1, y1


def align_crop(image_rgb, p_a, p_b, size=256, frac_a=0.35, frac_b=0.65,
               scale=None, centre=None):
    """
    Pose-normalise a specimen using two predicted keypoints.

    Builds a similarity transform (rotation + uniform scale + translation) that puts
    the p_a -> p_b body axis vertical, so rotation stops being a nuisance variable
    before ViT training.  Where the *scale* comes from is the caller's choice:

    keypoints mode (scale=None, the original behaviour)
        Sends p_a -> (size/2, size*frac_a) and p_b -> (size/2, size*frac_b), so the
        anchor axis always measures size*(frac_b - frac_a) px:

            s = |q_b - q_a| / |p_b - p_a|

        Beware: s is inversely proportional to the predicted anchor distance, which
        is neither proportional to body size (it depends on pose and on which
        landmarks the taxon actually exposes) nor stable (a low-confidence pair
        collapses towards one blob centre, shrinking |p_b - p_a| and blowing s up).
        The practical effect is that a specimen photographed *closer* has a larger
        anchor distance and so comes out *smaller* -- the framing gets inverted.

    box mode (scale given)
        s is passed in, computed from the detected specimen extent, and only the
        rotation is taken from the keypoints.  The output is centred on `centre`
        (the box centre) rather than pinned through the anchors.  Apparent specimen
        size is then the same for every image regardless of shot distance, and a
        degenerate anchor pair costs a wrong rotation instead of a wild zoom.

        theta = atan2(q_b - q_a) - atan2(p_b - p_a)
        M     = [[s*cos, -s*sin, tx],
                 [s*sin,  s*cos, ty]],   t = q_centre - R @ centre

    Returns a (size, size, 3) RGB array, or None if the two points coincide.
    """
    q_a = np.array([size / 2.0, size * frac_a])
    q_b = np.array([size / 2.0, size * frac_b])

    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    d = p_b - p_a
    e = q_b - q_a
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-6:
        return None

    s = np.linalg.norm(e) / norm_d if scale is None else float(scale)
    theta = math.atan2(e[1], e[0]) - math.atan2(d[1], d[0])
    cos_t, sin_t = s * math.cos(theta), s * math.sin(theta)

    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    # Anchor the translation on the box centre in box mode, on p_a otherwise: with an
    # externally supplied s the anchors no longer land on q_a/q_b, so pinning through
    # them would push the specimen off-frame by the scale mismatch.
    if scale is None:
        t = q_a - R @ p_a
    else:
        src = np.asarray(centre if centre is not None else (p_a + p_b) / 2.0,
                         dtype=np.float64)
        t = np.array([size / 2.0, size / 2.0]) - R @ src
    M = np.hstack([R, t.reshape(2, 1)])

    return cv2.warpAffine(image_rgb, M, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def letterbox(image_rgb, size):
    """
    Fit a crop onto a size x size canvas, aspect preserved, centred, zero-padded.

    Used only for specimens that could not be aligned (see --align-fallback), so that
    every file in the export directory still has the same geometry.
    """
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(size / w, size / h)
    resized = cv2.resize(image_rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]

    canvas = np.zeros((size, size, 3), dtype=image_rgb.dtype)
    top, left = (size - rh) // 2, (size - rw) // 2
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(root):
    """
    Walk `root`, returning image paths *relative* to it so the outputs can mirror the
    input tree.

    Class-labelled corpora keep specimens in per-class subdirectories, and the label
    lives only in the directory name -- so flattening the export would throw it away.
    Worse, two classes may each contain an `001.jpg`, and a flat export would have one
    silently overwrite the other.
    """
    # os.walk swallows a missing directory and simply yields nothing, which would leave the
    # whole run a silent no-op.  Check explicitly so a mistyped --input fails loudly.
    if not os.path.isdir(root):
        raise SystemExit(f"[error] --input is not a directory: {os.path.abspath(root)}")

    found = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(found)


def load_model(checkpoint_path, device):
    """
    Load the CPM checkpoint, spanning the torch versions this repo gets run under.

    PyTorch 2.6 flipped `weights_only` to True, which refuses this checkpoint because it
    carries numpy scalars -- fine to unpickle, since the file is produced by train.py in
    this repo rather than fetched from anywhere.  But the keyword did not exist before
    1.13, and the pinned insect_model environment is on 1.12, so passing it
    unconditionally would break the very environment the guide tells you to use.  Ask the
    installed torch which it is.
    """
    model = CPM(N_POINTS).to(device).eval()

    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(checkpoint_path, **load_kwargs)["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    return model


def draw(image_bgr, coords, scores, box=None):
    out = image_bgr.copy()
    radius = max(2, int(round(min(out.shape[:2]) / 200)))
    thickness = max(1, radius // 2)

    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    for i, ((x, y), s) in enumerate(zip(coords, scores)):
        x, y = int(round(x)), int(round(y))
        cv2.circle(out, (x, y), radius * 2, (255, 0, 0), -1)
        cv2.putText(out, f"{KP_NAMES[i]} {s:.2f}", (x + radius * 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, radius / 6.0, (255, 0, 0), thickness, cv2.LINE_AA)
    return out


def select_device(requested, strict=False):
    """
    Resolve --device, degrading to CPU when CUDA is present but not actually usable.

    torch.cuda.is_available() only reports that a driver and device were enumerated.
    On a shared cluster the GPU is often in Exclusive_Process compute mode and already
    claimed by another job, so it reports True and the refusal ("CUDA-capable device(s)
    is/are busy or unavailable") surfaces later, on the first real allocation.  Probe
    with a one-element tensor here so the fallback happens before the checkpoint load.
    """
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        reason = "no CUDA device visible"
    else:
        try:
            torch.zeros(1, device=requested)
            return torch.device(requested)
        except (RuntimeError, AssertionError) as exc:
            reason = str(exc).splitlines()[0]

    if strict:
        raise SystemExit(f"[error] {requested} unusable: {reason}")
    print(f"[warn] {requested} unusable ({reason}); falling back to CPU")
    return torch.device("cpu")


def main(args):
    device = select_device(args.device, strict=args.strict_device)
    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.checkpoint, device)
    to_tensor = transforms.ToTensor()

    # The centre map is a fixed gaussian at the image centre -- identical for every
    # sample in training, so build it once.
    centermap = gaussian(TARGET_SIZE // 2, TARGET_SIZE // 2, TARGET_SIZE, TARGET_SIZE)
    centermap = torch.from_numpy(centermap)[None, None].to(device)

    manual_boxes = {}
    if args.box_mode == "file":
        with open(args.boxes, "r") as handle:
            manual_boxes = json.load(handle)

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    align_a, align_b = (KP_NAMES.index(f"kp{v}") for v in args.align_pair.split(","))

    files = find_images(args.input)
    if not files:
        raise SystemExit(f"[error] no images under {os.path.abspath(args.input)} "
                         f"(searched recursively for {', '.join(IMAGE_EXTS)})")
    print(f"Found {len(files)} images under {os.path.abspath(args.input)}")

    exported = []
    rows = []
    for relpath in tqdm(files, desc="Inference"):
        path = os.path.join(args.input, relpath)
        # Everything downstream mirrors `subdir`, so an input tree of class folders comes
        # back out as the same tree of class folders.  `filename` keeps the relative path
        # in POSIX form: it is what the CSV and the log lines identify an image by, and
        # forward slashes keep those readable on either platform.
        subdir, name = os.path.split(relpath)
        filename = relpath.replace(os.sep, "/")

        pil_image = Image.open(path).convert("RGB")
        image_np = np.asarray(pil_image)
        h, w = image_np.shape[:2]

        # box_side is the specimen extent used to scale the aligned crop.  It tracks the
        # box the detector actually wanted, before clipping to the frame, so specimens
        # running off the edge are not scaled down by the amount that got cut away.
        if args.box_mode == "file":
            # Accept either the relative path (in whichever slash style the JSON was
            # written with) or a bare basename, so hand-written box files keyed on plain
            # filenames keep working against a nested input tree.
            box = next((manual_boxes[k] for k in (filename, relpath, name)
                        if k in manual_boxes), None)
            if box is None:
                print(f"  [skip] no box for {filename}")
                continue
            box_side = max(box[2] - box[0], box[3] - box[1])
        elif args.box_mode == "auto":
            detected = detect_box(image_np, margin=args.margin)
            if detected is None:
                print(f"  [warn] no insect found in {filename}, falling back to full frame")
                box, box_side = (0.0, 0.0, float(w), float(h)), float(max(w, h))
            else:
                box, box_side = detected
        else:
            box, box_side = (0.0, 0.0, float(w), float(h)), float(max(w, h))

        canvas, scale, ox, oy = preprocess(pil_image, box)

        with torch.no_grad():
            tensor = to_tensor(Image.fromarray(canvas)).unsqueeze(0).to(device)
            preds = model(tensor, centermap)
            final = preds[-1].cpu()          # last CPM stage, shape (1, 6, 28, 28)
            final = final[:, 1:]             # drop the background channel

        coords224 = heatmap2coord(final, topk=args.topk).numpy()[0]
        scores = final[0].flatten(1).max(dim=1).values.numpy()

        # 224-space -> crop-space -> original image pixels
        coords = coords224 / scale
        coords[:, 0] += ox
        coords[:, 1] += oy

        # The overlay is a full-resolution re-encode of the source frame, so it dominates both
        # runtime and disk once the corpus is large.  --no-vis drops it; keypoints.csv still
        # records every coordinate, so nothing quantitative is lost.
        if not args.no_vis:
            vis = draw(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), coords, scores,
                       box=box if args.draw_box else None)
            vis_dir = os.path.join(args.output, subdir)
            os.makedirs(vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, f"pred_{name}"), vis)

        stem = os.path.splitext(name)[0]
        status = ""

        # Specimen crop, pose-normalised: the similarity transform crops, rotates and
        # rescales in one step, so this is both the crop and the alignment.  It is the
        # input for MAE pre-training, which applies its own RandomResizedCrop and so
        # wants --align-size > 224 for headroom.
        # Alignment needs both anchor keypoints, so specimens the model is unsure about
        # are either skipped or exported unaligned, per --align-fallback.
        if args.save_crops:
            crop = None
            reason = None
            if min(scores[align_a], scores[align_b]) < args.align_min_score:
                reason = (f"anchor score below {args.align_min_score} "
                          f"({scores[align_a]:.2f}, {scores[align_b]:.2f})")
            else:
                if args.align_scale_mode == "box":
                    box_scale = args.align_size * args.align_fill / max(box_side, 1e-6)
                    box_centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                else:
                    box_scale, box_centre = None, None
                crop = align_crop(image_np, coords[align_a], coords[align_b],
                                  size=args.align_size,
                                  frac_a=0.5 - args.align_span / 2.0,
                                  frac_b=0.5 + args.align_span / 2.0,
                                  scale=box_scale, centre=box_centre)
                if crop is None:
                    reason = "anchor keypoints coincide"

            status = "aligned"
            if reason is not None:
                if args.align_fallback == "crop":
                    cx1, cy1, cx2, cy2 = [int(round(v)) for v in box]
                    crop = letterbox(image_np[cy1:cy2, cx1:cx2], args.align_size)
                    status = "unaligned"
                    print(f"  [unaligned] {filename}: {reason}")
                else:
                    status = "skipped"
                    print(f"  [skip] {filename}: {reason}")

            if crop is not None:
                crop_dir = os.path.join(args.save_crops, subdir)
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f"{stem}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                exported.append(os.path.abspath(crop_path))

        row = {"filename": filename, "box": " ".join(f"{v:.1f}" for v in box),
               "crop": status}
        for kp_name, (x, y), s in zip(KP_NAMES, coords, scores):
            row[f"{kp_name}_x"] = round(float(x), 2)
            row[f"{kp_name}_y"] = round(float(y), 2)
            row[f"{kp_name}_score"] = round(float(s), 4)
        rows.append(row)

    if args.list_file and exported:
        with open(args.list_file, "w") as handle:
            handle.write("\n".join(exported) + "\n")
        print(f"Wrote {len(exported)} paths to {args.list_file}")

    if rows:
        csv_path = os.path.join(args.output, "keypoints.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/images/inference")
    parser.add_argument("--output", default="output/inference")
    parser.add_argument("--checkpoint", default="epoch-latest.pth")
    parser.add_argument("--box-mode", choices=["auto", "full", "file"], default="auto")
    parser.add_argument("--boxes", default="boxes.json", help="used with --box-mode file")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="fraction to expand the auto-detected box by")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-device", action="store_true",
                        help="fail instead of falling back to CPU when the GPU is unusable")
    parser.add_argument("--draw-box", action="store_true", help="draw the crop box on outputs")
    parser.add_argument("--no-vis", action="store_true",
                        help="skip the pred_*.jpg overlays entirely; keypoints.csv is still "
                             "written to --output.  Use with --save-crops when you only want "
                             "the exported crops")

    # Exports for downstream ViT training
    parser.add_argument("--save-crops", default=None, metavar="DIR",
                        help="write pose-normalised specimen crops (MAE input)")
    parser.add_argument("--align-pair", default="4,8",
                        help="the two keypoint IDs defining the body axis, e.g. 4,8")
    parser.add_argument("--align-size", type=int, default=256,
                        help="side length of the crop; keep > 224 for RandomResizedCrop")
    parser.add_argument("--align-scale-mode", choices=["box", "keypoints"], default="box",
                        help="what sets the zoom: the detected specimen box (consistent "
                             "apparent size across shot distances) or the anchor keypoint "
                             "distance (legacy; inverts framing and is noise-sensitive)")
    parser.add_argument("--align-fill", type=float, default=0.95,
                        help="with --align-scale-mode box, fraction of the output side "
                             "spanned by the specimen box; lower zooms out")
    parser.add_argument("--align-span", type=float, default=0.30,
                        help="with --align-scale-mode keypoints, fraction of the output "
                             "height spanned by the anchor axis; lower zooms out")
    parser.add_argument("--align-min-score", type=float, default=0.3,
                        help="treat a specimen as unalignable if either anchor scores below this")
    parser.add_argument("--align-fallback", choices=["skip", "crop"], default="skip",
                        help="what to export for unalignable specimens: nothing, or the "
                             "plain box crop letterboxed to --align-size")
    parser.add_argument("--list-file", default=None, metavar="PATH",
                        help="write exported image paths for ImageDatasetListV2")
    main(parser.parse_args())
