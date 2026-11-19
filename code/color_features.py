import numpy as np
from skimage.color import rgb2hsv


def quantized_hsv(rgb, bins):
    hsv = rgb2hsv(rgb)
    return [
        np.minimum((hsv[:, :, channel] * count).astype(np.int16), count - 1)
        for channel, count in enumerate(bins)
    ]


def color_histogram(rgb, payload):
    settings = payload["metadata"]["color_features"]
    grid_y, grid_x = settings["grid"]
    height, width = rgb.shape[:2]
    if height % grid_y or width % grid_x:
        raise ValueError("Color feature windows must be divisible by the spatial grid")
    channels = quantized_hsv(rgb, settings["bins"])
    features = []
    for row in range(grid_y):
        for column in range(grid_x):
            top, bottom = row * height // grid_y, (row + 1) * height // grid_y
            left, right = column * width // grid_x, (column + 1) * width // grid_x
            for channel, count in zip(channels, settings["bins"]):
                histogram = np.bincount(channel[top:bottom, left:right].ravel(), minlength=count)
                features.extend(histogram.astype(np.float32) / ((bottom - top) * (right - left)))
    return np.asarray(features, dtype=np.float32)
