import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
import pycocotools.mask as maskUtils
from skimage import measure
import cv2

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import pdb

class RobotSegDataset(object):
    def __init__(self, image_dir, ann_dir, transforms=None):
        self.ann_dir = ann_dir
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transforms = transforms
         
    def __len__(self):
        return len(self.image_names)
    
    def get_binary(self, mask):
        bw_masks = []
        classes = np.unique(mask)
        for c in classes:
            bw_masks.append((mask == c).astype(np.uint8))
        return bw_masks, classes

    def to_polygon(self, mask):
        poly = []
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            poly.append(segmentation)
        return poly
    
    def __getitem__(self, idx):
        # load the image as a PIL Image
        im_name = self.image_names[idx]
        image = Image.open(osp.join(self.image_dir, im_name))
        # load annotations masks 
        ann = np.array(Image.open(osp.join(self.ann_dir, im_name)))
        bw_masks, labels = self.get_binary(ann)        
        # convert masks to rles and polygons
        bboxes = []
        poly_masks = []
        rle_masks = []
        for mask in bw_masks:
            # get polygons for each mask
            poly_masks.append(self.to_polygon(mask))
            current_mask = maskUtils.encode(np.asfortranarray(mask))
            rle_masks.append(current_mask)
            # get bboxes with pycocotools.mask.toBbox(rle) as a list of lists
            #bboxes.append(cv2.boundingRect(mask))
            bboxes.append(maskUtils.toBbox(current_mask))
                    
        # create a BoxList from the boxes
        target = BoxList(bboxes, image.size, mode="xyxy") #xywh
        # add the labels to the target annotations
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        # add masks to the target annotations
        masks = SegmentationMask(poly_masks, image.size)
        target.add_field("masks", masks)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # return the image, the boxlist and the idx in your dataset
        return image, target, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        im_name = self.image_names[idx]
        image = Image.open(osp.join(self.image_dir, im_name))
        img_width, img_height = image.size
        return {"height": img_height, "width": img_width}