import cv2
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
        dirnames: List[str] = None,
        image_size: int = 416,
        image_patterns: List[str] = ['*.jpg'],
        label_patterns: List[str] = ['*.json'],
        classes: Dict[str, int] = None,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        transforms: Optional[List] = None,
    ) -> None:
        super(YOLODataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)       # convert std and mean to tensor then reshape to math with shape of input tensor
        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.pad_to_square = iaa.PadToSquare(position='right-bottom')       # pad 0 values to input image to shape is square. And just pad for right or bottom  
        self.transforms = transforms if transforms else []

        image_paths, label_paths = [], []
        for dirname in dirnames:
            for image_pattern in image_patterns:
                image_paths.extend(Path(dirname).glob('**/{}'.format(image_pattern)))       # add path of all images to image_paths list
            for label_pattern in label_patterns:
                label_paths.extend(Path(dirname).glob('**/{}'.format(label_pattern)))       # add path of all labels to image_paths list

        image_paths = natsorted(image_paths, key=lambda x: str(x.stem))                     # sort list follow name
        label_paths = natsorted(label_paths, key=lambda x: str(x.stem))

        self.data_pairs = [[image, label] for image, label in zip(image_paths, label_paths) if image.stem == label.stem]        # zip each other image and label to data_pair list


    def __len__(self):
        return len(self.data_pairs)

    def _get_label_info(self, lable_path: str, w: int, h: int) -> Dict:          # this function take information of label (pre-process label).
        bboxes = np.roll(np.loadtxt(fname=lable_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        bboxes = [[
                   (box[0] - box[2] / 2) * w,
                   (box[1] - box[3] / 2) * h,
                   (box[0] + box[2] / 2) * w,
                   (box[1] + box[3] / 2) * h,
                   box[4]
        ] for box in bboxes]

        label_info = []
        for box in bboxes:
            label_info.append({'label': box[4], 'bbox': (box[0], box[1], box[2], box[3])})

        if not len(label_info):         # Neu image k co doi tuong da dc label nao thi tra ve label: -1 va k co box 
            label_info.append({'label': -1, 'bbox': (0, 0, 1, 1)})

        return label_info       # tra ve ten va box cua tung doi tuong duoc label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Tuple[str, Tuple[int, int]]]:
        image_path, label_path = self.data_pairs[idx]
        image = cv2.imread(str(image_path))                 # Doc file anh
        image_info = (str(image_path), image.shape[1::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # Chuyen image tu dang BGR sang RGB (dang chuan cua image)
        h, w = image.shape[:2]
        label_info = self._get_label_info(lable_path=str(label_path), w=w, h=h)

        if len(label_info) == 1 and label_info[0]['label'] == -1:       
        # Neu roi vao TH anh k co doi tuong nao dc label => tra ve frame binh thuong
            # Target
            target = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'boxes': torch.tensor([[0., 0., 1., 1.]], dtype=torch.float32),
                'labels': torch.tensor([-1], dtype=torch.int64),
            }

            # Sample
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))
            sample = torch.from_numpy(np.ascontiguousarray(image))
            sample = sample.permute(2, 0, 1).contiguous()
            sample = (sample.float().div(255.) - self.mean) / self.std

            return sample, target, image_info


        boxes = [label['bbox'] for label in label_info]         # lay boxes: toa do cua box
        labels = [label['label'] for label in label_info]       # lay labels: 0 hoac 1 (de sau se chuyen ve mark(0) hoac no_mark(1))

        bbs = BoundingBoxesOnImage([BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=label)
                                    for box, label in zip(boxes, labels)], shape=image.shape)       # Gan boxes len image 
        for transform in random.sample(self.transforms, k=random.randint(0, len(self.transforms))):
            image, bbs = transform(image=image, bounding_boxes=bbs)         # Qua cac phep transform (giup data da dang hon)

        image, bbs = self.pad_to_square(image=image, bounding_boxes=bbs)        # Pad thanh hinh vuong de giu ti le cua doi tuong
        image, bbs = iaa.Resize(size=self.image_size)(image=image, bounding_boxes=bbs)      # chuyen anh ve size 416x416
        bbs = bbs.on(image)

        # Chuyen ve dang list cua boxes and labels
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]     # lay boxes cuoi cung sau khi da xu ly xong
        labels = [bb.label for bb in bbs.bounding_boxes]                        # lay labels

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)  # 
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # thong tin cua cac boxes 
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }

        # chuyen image tu dang array ve dang tensor
        sample = torch.from_numpy(np.ascontiguousarray(image))
        sample = sample.permute(2, 0, 1).contiguous()
        sample = (sample.float().div(255.) - self.mean) / self.std

        return sample, target, image_info
