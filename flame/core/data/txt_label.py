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

# ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        img_path,
        anno_path,
        anchors,
        mean,
        std,
        image_size=416,
        S=[13, 26, 52],
        C=13,
        transforms=None,
    ):
        self.img_path = img_path
        self.anno_path = anno_path
        self.image_size = image_size
        self.transforms = transforms
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    

    def make_datapath_list(self):
        img_paths = sorted(Path(self.img_path).glob(f"*.jpg"))
        anno_paths = sorted(Path(self.anno_path).glob(f"*.txt"))

        data_pairs = [[image_path, label_path] for image_path, label_path in zip(img_paths, anno_paths)]

        return data_pairs


    def iou(self, boxes1, boxes2):
        """
        Parameters:
            boxes1 (tensor): width and height of the first bounding boxes
            boxes2 (tensor): width and height of the second bounding boxes
        Returns:
            tensor: Intersection over union of the corresponding boxes
        """
        intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
            boxes1[..., 1], boxes2[..., 1]
        )
        union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
        )
        return intersection / union
    
    def __len__(self):
        return len(self.make_datapath_list())

    def __getitem__(self, idx):
        data_pairs = self.make_datapath_list()
        image_path, label_path = data_pairs[idx]
        bboxes = np.roll(np.loadtxt(fname=str(label_path), delimiter=" ", ndmin=2), 4, axis=1).tolist()
        image = np.array(Image.open(str(image_path)).convert("RGB"))
        image_info = (str(image_path), image.shape[1::-1])
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
