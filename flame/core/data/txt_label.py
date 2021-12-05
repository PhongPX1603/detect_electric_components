import cv2
import json
import torch
import random
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage



class YOLODataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_size,
        image_patterns,
        label_patterns,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        transforms=None):
        super(YOLODataset, self).__init__()
        self.image_size = image_size
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')
        self.transforms = transforms if transforms else []

    def make_datapath_list(self):
        img_paths = sorted(Path(self.img_path).glob(f"*.jpg"))
        anno_paths = sorted(Path(self.anno_path).glob(f"*.txt"))

        data_pairs = [[image_path, label_path] for image_path, label_path in zip(img_paths, anno_paths)]

        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        data_pairs = self.make_datapath_list()
        image_path, label_path = data_pairs[idx]
        bboxes = np.roll(np.loadtxt(fname=str(label_path), delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = cv2.imread(str(image_path))
        image_info = (str(image_path), image.shape[1::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        bboxes = [[
                   (box[0] - box[2] / 2) * w,
                   (box[1] - box[3] / 2) * h,
                   (box[0] + box[2] / 2) * w,
                   (box[1] + box[3] / 2) * h,
                   box[4]
        ] for box in bboxes]

        bounding_boxes = [BoundingBox(
                info[0],
                info[1],
                info[2],
                info[3],
                label=info[4]
            ) for info in bboxes]

        bounding_boxes = BoundingBoxesOnImage(bounding_boxes=bounding_boxes, shape=image.shape)

        for transform in self.transforms:
            image, bounding_boxes = transform(image=image, bounding_boxes=bounding_boxes)
        
        image, bounding_boxes = iaa.PadToSquare(position='right-bottom')(image=image, bounding_boxes=bounding_boxes)
        image, bounding_boxes = iaa.Resize(self.image_size)(image=image, bounding_boxes=bounding_boxes)

        bboxes = [[(bb.x1 + bb.x2) / (2 * self.image_size), 
                   (bb.y1 + bb.y2) / (2 * self.image_size), 
                   (bb.x2 - bb.x1) / self.image_size, 
                   (bb.y2 - bb.y1) / self.image_size] 
                  for bb in bounding_boxes]
        labels = [bb.label for bb in bounding_boxes]

        # Convert to Torch Tensor
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # suppose all instances are not crowd
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Boxes Info
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
