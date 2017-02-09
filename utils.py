import numpy as np
import scipy.misc


def grid_plot(images, size, path):
    images = (images + 1.0) / 2.0

    h, w = images.shape[1], images.shape[2]
    image_grid = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        image_grid[j * h:j * h + h, i * w:i * w + w, :] = image

    scipy.misc.imsave(path, image_grid)
