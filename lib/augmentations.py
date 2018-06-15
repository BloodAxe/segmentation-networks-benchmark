import numbers
import random
import torch

import cv2
import numpy as np
import math


class Sequential:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class OneOf:
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask


class OneOrOther:
    def __init__(self, first, second, prob=.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class MaskOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask):
        return x, self.trans(mask)


class RandomGrayscale():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


class RandomInvert:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            img = img.max() - img
        return img


class MakeBinary:
    def __call__(self, x):
        dt = x.dtype
        x = x > 0
        return x.astype(dt)


class VerticalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = np.flipud(img).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = np.fliplr(img).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        return img, mask


class Transpose:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2).copy()
            if mask is not None:
                mask = mask.transpose(1, 0).copy()
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor).copy()
            if mask is not None:
                mask = np.rot90(mask, factor).copy()
        return img, mask


class Rotate:
    def __init__(self, limit=90, prob=.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class Shift:
    def __init__(self, limit=4, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            limit = self.limit
            dx = round(random.uniform(-limit, limit))
            dy = round(random.uniform(-limit, limit))

            height, width, channel = img.shape
            y1 = limit + 1 + dy
            y2 = y1 + height
            x1 = limit + 1 + dx
            x2 = x1 + width

            img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1, borderType=cv2.BORDER_REFLECT_101)
            img = img1[y1:y2, x1:x2, :].copy()
            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit + 1, limit + 1, limit + 1, limit + 1, borderType=cv2.BORDER_REFLECT_101)
                mask = msk1[y1:y2, x1:x2, :].copy()

        return img, mask


class ShiftScale:
    def __init__(self, limit=4, prob=.25):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        limit = self.limit
        if random.random() < self.prob:
            height, width, channel = img.shape
            assert (width == height)
            size0 = width
            size1 = width + 2 * limit
            size = round(random.uniform(size0, size1))

            dx = round(random.uniform(0, size1 - size))
            dy = round(random.uniform(0, size1 - size))

            y1 = dy
            y2 = y1 + size
            x1 = dx
            x2 = x1 + size

            img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
            img = (img1[y1:y2, x1:x2, :] if size == size0
                   else cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

            if mask is not None:
                msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
                mask = (msk1[y1:y2, x1:x2, :] if size == size0
                        else cv2.resize(msk1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR))

        return img, mask


class ShiftScaleRotate:
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit, self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit, self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(img, mat, (width, height),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(mask, mat, (width, height),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class CenterCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2].copy()
        if mask is not None:
            mask = mask[y1:y2, x1:x2].copy()

        return img, mask


class RandomCrop(object):
    """Crop the given Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, x, mask=None):
        """
        Args:
            img: Image to be cropped.

        Returns:
            : Cropped image.
        """
        if self.padding > 0:
            x = np.pad(x, self.padding, 'constant')

        i, j, h, w = self.get_params(x, self.size)

        x = x[i:i + h, j:j + w].copy()

        if mask is not None:
            if self.padding > 0:
                mask = np.pad(mask, self.padding, 'constant')
            mask = mask[i:i + h, j:j + w].copy()

        return x, mask


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class RandomFilter:
    """
    blur sharpen, etc
    """

    def __init__(self, limit=.5, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = self.limit * random.uniform(0, 1)
            kernel = np.ones((3, 3), np.float32) / 9 * 0.2

            colored = img[..., :3]
            colored = alpha * cv2.filter2D(colored, -1, kernel) + (1 - alpha) * colored
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(colored, dtype, maxval)

        return img


# https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
# color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img


class RandomContrast:
    def __init__(self, limit=.1, prob=.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)
        return img


class RandomSaturation:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        # dont work :(
        if random.random() < self.prob:
            maxval = np.max(img[..., :3])
            dtype = img.dtype
            alpha = 1.0 + random.uniform(-self.limit, self.limit)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
            img[..., :3] = clip(img[..., :3], dtype, maxval)
        return img


class RandomHueSaturationValue:
    def __init__(self, hue_shift_limit=(-10, 10), sat_shift_limit=(-25, 25), val_shift_limit=(-25, 25), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class NormalizeImage:
    def __init__(self, scale=1. / 255., mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.scale = float(scale)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x):
        x = (x * self.scale - self.mean) / self.std
        return x


class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


def tta_d4_aug(images):
    for image in images:
        yield image
        yield np.rot90(image, 1)
        yield np.rot90(image, 2)
        yield np.rot90(image, 3)

        yield np.fliplr(image)
        yield np.fliplr(np.rot90(image, 1))
        yield np.fliplr(np.rot90(image, 2))
        yield np.fliplr(np.rot90(image, 3))


def tta_d4_deaug(image_list):
    assert len(image_list) % 8 == 0
    for i in range(0, len(image_list), 8):
        img = np.zeros_like(image_list[i])

        img += image_list[i + 0]
        img += np.rot90(image_list[i + 1], -1)
        img += np.rot90(image_list[i + 2], -2)
        img += np.rot90(image_list[i + 3], -3)

        img += np.fliplr(image_list[i + 4])
        img += np.rot90(np.fliplr(image_list[i + 5]), -1)
        img += np.rot90(np.fliplr(image_list[i + 6]), -2)
        img += np.rot90(np.fliplr(image_list[i + 7]), -3)
        img /= 8.

        yield img
