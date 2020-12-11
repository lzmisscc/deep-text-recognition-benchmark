from imgaug.augmenters import weather
from imgaug.augmenters.meta import OneOf
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
import tqdm
import os.path as osp
import glob
import random

aug = [
    # iaa.OneOf([
    #     iaa.WithBrightnessChannels(),
    #     iaa.JpegCompression(compression=(0, 70)),
    # ]),
    iaa.OneOf([
        iaa.Dropout(p=(0.0, 0.08)),
        iaa.Dropout(p=(0, 0.08), per_channel=True),
        iaa.Dropout2d(),
        iaa.SaltAndPepper(),
        iaa.CoarseSaltAndPepper(),
        # iaa.JpegCompression(compression=(0, 10)),
        # iaa.AdditiveGaussianNoise(),
        # iaa.AdditiveGaussianNoise(per_channel=True),
        iaa.CoarseDropout(),
        iaa.CoarseDropout(per_channel=True),
        iaa.Multiply(),
    ]),
    iaa.OneOf([
        iaa.ChannelShuffle(),
    ]),
    # iaa.OneOf([
    #     iaa.GaussianBlur(sigma=(0.0, 1.1)),
    #     iaa.AverageBlur(k=(1, 3)),
    #     iaa.MotionBlur(k=(3, 5)),
    # ]),
    iaa.OneOf([
        iaa.PerspectiveTransform(scale=(0, 0.01)),
    ]),
    # iaa.OneOf([
    #     weather.FastSnowyLandscape(lightness_multiplier=(0, 5)),
    #     weather.Rain(),
    #     weather.Snowflakes(),
    # ])
]


aug = iaa.Sometimes(
    0.8,
    iaa.SomeOf(
        (1,3),
        aug
    )
)


aug = iaa.Sequential(aug)