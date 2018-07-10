import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic, join_segmentations, quickshift
from skimage.morphology import watershed
from skimage.color import label2rgb
from sklearn import cluster
from sklearn.feature_extraction import image as image_extract


def segment_sobel_watershed(image):
    # Make segmentation using edge-detection and watershed.
    edges = sobel(image)

    show(edges, 'sobel')

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.ones_like(image) / 2
    foreground, background = 0, 1

    markers[image < 0.3] = background
    markers[image > 0.7] = foreground
    show(image, 'image')
    show(markers, 'markers')

    ws = watershed(edges, markers)

    return label(ws == foreground)


def segment_slic(image):
    # Make segmentation using SLIC superpixels
    return slic(image, n_segments=8,
            multichannel=False)


def segment_quickshift(image):
    return quickshift(image.permute(1, 2, 0), ratio=0.6, kernel_size=5, max_dist=10, return_tree=False, sigma=0.2)

def show(image, title='default'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    from skimage import data

    image = data.coins()
    seg1 = segment_sobel_watershed(image)
    seg2 = segment_slic(image)

    # Combine the two.
    segj = join_segmentations(seg1, seg2)

    # Show the segmentations.
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 5),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Image')

    color1 = label2rgb(seg1, image=image, bg_label=0)
    ax[1].imshow(color1)
    ax[1].set_title('Sobel+Watershed')

    color2 = label2rgb(seg2, image=image, image_alpha=0.5)
    ax[2].imshow(color2)
    ax[2].set_title('SLIC superpixels')

    color3 = label2rgb(segj, image=image, image_alpha=0.5)
    ax[3].imshow(color3)
    ax[3].set_title('Join')

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()