"""
Write an OpenCV program to do the following things:

    1. Read an image from a file and display it to the screen
    2. Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
    3. Resize the image uniformly by half
"""

import cv2
import numpy as np


def display_image(img, win_name):
    """
        img : Input image to display
        win_name : Title of the window
    """
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    # Not Destroying windows for comparison


def resize_image(img, file_name, fx=0.8, fy=0.8, rewrite=False):
    """
        img : Input image to resize
        file_name : file name in case there is a need to overwrite
        fx : scale factor along the horizontal axis
        fy : scale factor along the vertical axis
        rewrite : rewrite flag
    """
    img = cv2.resize(img, None, fx=fx, fy=fy)

    if rewrite == True:
        cv2.imwrite(file_name, img)

    return img


# Set window names
raw_img_win = "raw_image"
scalar_mod_img_win = "scalar_mod_image"
resize_img_win = "resize_image"
file_name = "NewYorkSkyline.jpg"

# Read input image
img = np.array(cv2.imread(file_name))

# Check the dimensions of the image
height, width = img.shape[:2]
print width, height

if height > 1500 or width > 900:
    img = resize_image(img, file_name, True)

display_image(img, raw_img_win)  #display original image

new_img = np.array([row + 50 for row in img])
display_image(new_img,
              scalar_mod_img_win)  #display after adding 50 to each pixel

new_img = np.array([row - 50 for row in img])
display_image(new_img,
              scalar_mod_img_win)  #display after subtracting 50 to each pixel

new_img = np.array([row * 2 for row in img])
display_image(new_img,
              scalar_mod_img_win)  #display after multiplying each pixel by 2

new_img = np.array([row / 2 for row in img])
display_image(new_img,
              scalar_mod_img_win)  #display after dividing each pixel by 2

new_img = resize_image(img, file_name, 0.5, 0.5)
display_image(new_img, resize_img_win)  #display after resizing the image

# Destroy all windows
cv2.destroyAllWindows()
