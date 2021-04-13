import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.image_list import ImageList

class NothingTransform(nn.Module):
    def __init__(self):
        super(NothingTransform, self).__init__()

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        # images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            # image = self.normalize(image)
            # image, target_index = self.resize(image, target_index)
            # images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # image_sizes = [img.shape[-2:] for img in images]
        # images = self.batch_images(images)
        # image_sizes_list: List[Tuple[int, int]] = []
        # for image_size in image_sizes:
        #     assert len(image_size) == 2
        #     image_sizes_list.append((image_size[0], image_size[1]))

        image_sizes_list = [(images.shape[2], images.shape[3])]*images.shape[0]
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


    def postprocess(self,
                    result,               # type: List[Dict[str, Tensor]]
                    image_shapes,         # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            #boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        #     if "masks" in pred:
        #         masks = pred["masks"]
        #         masks = paste_masks_in_image(masks, boxes, o_im_s)
        #         result[i]["masks"] = masks
        #     if "keypoints" in pred:
        #         keypoints = pred["keypoints"]
        #         keypoints = resize_keypoints(keypoints, im_s, o_im_s)
        #         result[i]["keypoints"] = keypoints
        return result