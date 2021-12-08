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
        self.pad_to_square = iaa.PadToSquare(position='center')
        self.transforms = transforms if transforms else []

        img_paths = sorted(Path(image_dir).glob("*.jpg"))
        anno_paths = sorted(Path(label_dir).glob("*.txt"))

        self.data_pairs = [[image_path, label_path] for image_path, label_path in zip(img_paths, anno_paths)]
        
        print(len(self.data_pairs))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]
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

        bbs = BoundingBoxesOnImage(bounding_boxes=bounding_boxes, shape=image.shape)

        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)         # implement transforms (data augementation)

        # Rescale image and bounding boxes
        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)        # pad 0 values to input image to shape is square. And just pad for right or bottom 
        image, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)      # resize to input size of sample
        bbs = bbs.on(image)

        # Convert from Bouding Box Object to boxes, labels list
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]     # take bboxes after transforms
        labels = [bb.label for bb in bbs.bounding_boxes]

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
