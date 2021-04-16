from skimage.exposure import rescale_intensity
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
import numpy as np
import cv2

def image_convolution(image, kernel):

    # Define dimensions of image and kernel
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    # Add padding around input image
    padding = (kernel_width - 1) // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    
    # Allocate memory for output image
    image_output = np.zeros((image_height, image_width), dtype = "float32")

    for y in np.arange(padding, image_height + padding):
        for x in np.arange(padding, image_width + padding):

            # Extract region of interest (ROI)
            roi = image[y - padding : y + padding + 1, x - padding : x + padding + 1]

            # Perform convolution
            k = (roi * kernel).sum()

            # Store the convolution value in the output image
            image_output[y - padding, x - padding] = k

    # Rescale output image to [0, 255] range
    image_output = rescale_intensity(image_output, in_range = (0, 255))
    image_output = (image_output * 255).astype("uint8")

    # Return the output image
    return image_output


def image_filter(filter):

    if filter == 'sharpen':
        image_kernel = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype = "int")

        return image_kernel

    elif filter == 'laplacian':
        image_kernel = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype = "int")

        return image_kernel

    else:
        raise AttributeError('Unknown. Please specify a valid filter.')


def filter_wiener(image):

        psf = np.ones((5, 5)) / 125
        image = conv2(image, psf, 'same')
        image -= 0.3 * image.std() * np.random.standard_normal(image.shape)

        deconvolved, _ = restoration.unsupervised_wiener(image, psf, clip = False)

        return deconvolved / 255


def image_read(img):

    # Load image and convert to grayscale
    image = cv2.imread('3d_pokemon.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


