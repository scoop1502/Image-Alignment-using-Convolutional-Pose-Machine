import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision
import cv2
import random

class KPDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.image_dir = f"{self.root}/images/default"
        with open(f"{self.root}/kp_data.pkl", "rb") as handle:
            data = pickle.load(handle)
            self.images = data["images"]
            self.boxes = data["boxes"]
            self.points = data["points"]
            skip_list = sorted(data["skip_list"])[::-1]
            for i in skip_list:
                self.images.pop(i)
                self.boxes.pop(i)
                self.points.pop(i)
        self.insects = []
        for filename in self.images:
            filename = os.path.basename(filename)
            self.insects.append(filename.split("_")[0])
        self.insect_classes = sorted(list(set(self.insects)))
        self.split = split

        self.kp_6map8 = {
            1: 1,
            2: 2,
            3: 4,
            4: 6,
            5: 5,
            6: 8
        }

        self.target_size = 224

        self.indices = []

        for insect_class in self.insect_classes:
            indices = [i for i in range(len(self.images)) if self.insects[i] == insect_class]
            frontal = [i for i in indices if len(self.points[i]) == 8]
            profile = [i for i in indices if len(self.points[i]) == 6]
            assert len(indices) == len(frontal) + len(profile), insect_class
            n1 = int(0.8 * len(frontal))
            n2 = int(0.8 * len(profile))
            if split == "train":
                frontal = frontal[:n1]
                profile = profile[:n2]
            else:
                frontal = frontal[n1:]
                profile = profile[n2:]
            self.indices = self.indices + frontal # + profile

            print(f"{insect_class}. frontal: {len(frontal)}. profile: {len(profile)}")



        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        filename = self.images[index]
        points = self.points[index]
        cls = 1 if len(points) == 8 else 0
        image_path = f"{self.image_dir}/{filename}"
        x1, y1, x2, y2 = [float(v) for v in self.boxes[index][0]]
        #print(x1, y1, x2, y2)
        lmks = np.zeros((8, 2), dtype=np.float32)

        for p in points:
            x = float(p[0])
            y = float(p[1])
            k = int(p[2])
            if len(points) == 6:
                k = self.kp_6map8[k]
            lmks[k - 1, 0] = x
            lmks[k - 1, 1] = y


        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        image = image.crop([x1, y1, x2, y2])
        w, h = image.size
        lmks[:, 0] = (lmks[:, 0] - x1) / w
        lmks[:, 1] = (lmks[:, 1] - y1) / h

        scale_x = self.target_size / w
        scale_y = self.target_size / h
        scale = min(scale_x, scale_y)
        image = np.asarray(image)
        image = cv2.resize(image, None, fx=scale, fy=scale)

        #print(image.shape)
        h, w,_ = image.shape
        box_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        box_image[:h, :w, :] = image
        lmk_mask = np.ones((8, 1), dtype=np.float32)
        if len(points) == 6:
            lmk_mask[2] = 0
            lmk_mask[6] = 0

        if self.split == "train":
            if random.random() >= 0.5: ## FLIP X
                box_image = box_image[:, ::-1, :]
                lmks[:, 0] = 1 - lmks[:, 0]
            if random.random() >= 0.5: ## FLIP Y
                box_image = box_image[::-1, :, :]
                lmks[:, 1] = 1 - lmks[:, 1]

        image = Image.fromarray(box_image)
        image = self.transform(image)

        return image, cls, lmks, lmk_mask

if __name__ == "__main__":
    dataset =  KPDataset("data")

    image, cls, lmks, lmk_mask = dataset[0]

    dataset = KPDataset("data", split="test")


