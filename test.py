import numpy as np
import scipy.ndimage
from tifffile import tifffile



video = tifffile.imread("data/ca-cell.tif")
print(video.shape)
print(video.dtype)
print(video.min())
print(video.max())
print(video.mean())
print(video.std())
print(video.var())
print(video.sum())

ca = np.load("data/ca.npy")
print(ca.shape)
