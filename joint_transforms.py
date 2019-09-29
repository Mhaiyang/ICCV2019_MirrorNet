"""
 @Time    : 9/29/19 21:16
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : joint_transforms.py
 @Function: transform functions.
 
"""
import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomRotate(object):
    def __call__(self, img, mask):
        p = random.random()
        # rotate
        # if p < 0.25:
        #     return img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90)
        # if p < 0.5:
        #     return img.transpose(Image.ROTATE_180), mask.transpose(Image.ROTATE_180)
        # if p < 0.75:
        #     return img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270)

        # flip
        if p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, img, mask, edge):
#         assert img.size == mask.size
#         assert img.size == edge.size
#         for t in self.transforms:
#             img, mask, edge = t(img, mask, edge)
#         return img, mask, edge
#
#
# class RandomRotate(object):
#     def __call__(self, img, mask, edge):
#         p = random.random()
#         # rotate
#         # if p < 0.25:
#         #     return img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90), edge.transpose(Image.ROTATE_90)
#         # if p < 0.5:
#         #     return img.transpose(Image.ROTATE_180), mask.transpose(Image.ROTATE_180), edge.transpose(Image.ROTATE_180)
#         # if p < 0.75:
#         #     return img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270), edge.transpose(Image.ROTATE_270)
#
#         # # flip
#         if p < 0.5:
#             return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
#                    edge.transpose(Image.FLIP_LEFT_RIGHT)
#
#         return img, mask, edge
#
#
# class Resize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)
#
#     def __call__(self, img, mask, edge):
#         assert img.size == mask.size
#         assert img.size == edge.size
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), \
#                edge.resize(self.size, Image.NEAREST)
