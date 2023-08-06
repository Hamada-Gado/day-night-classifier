import cv2
import numpy as np

SHAPE = (1080, 1920, 3)

Image = np.ndarray[np.uint8, np.dtype[np.uint8]]

# Constants
root_path = ".."
images_path = f"{root_path}/databases/intermittent"
day_images_path = f"{images_path}/day"
night_images_path = f"{images_path}/night"


# 
## noise removal
def remove_noise_medianblur(image: Image) -> Image:
    return cv2.medianBlur(image, 5)

def remove_noise_gaussianblur(image: Image) -> Image:
    return cv2.GaussianBlur(image, (5, 5), 0)

def remove_noise_fNlMean(image: Image) -> Image:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

##thresholding
def thresholding(image: Image) -> Image:
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

## canny edge detection
def canny(image: Image) -> Image:
    return cv2.Canny(image, 100, 200)