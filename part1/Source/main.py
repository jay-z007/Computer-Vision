# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    res = []
    height, width = img_in.shape[:2]
    split_img = cv2.split(img_in)

    for i, channel in enumerate(split_img):
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = np.cumsum(hist)
        cdf = cdf * 255 / (height * width)

        res.append(cdf[channel])

    img_out = cv2.merge(res)

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):

    # Write low pass filter here
    res = []
    height, width = img_in.shape[:2]
    split_img = cv2.split(img_in)

    for i, channel in enumerate(split_img):
        f_img_in = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_img_in)

        rows, cols = channel.shape
        crow, ccol = rows / 2, cols / 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - 20:crow + 20, ccol - 20:ccol + 20] = 1
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        res.append(img_back)

    img_out = cv2.merge(res)  # Low pass filter result

    return True, img_out


def high_pass_filter(img_in):

    # Write high pass filter here
    res = []
    height, width = img_in.shape[:2]
    split_img = cv2.split(img_in)

    for i, channel in enumerate(split_img):
        f_img_in = np.fft.fft2(channel)
        f_shift = np.fft.fftshift(f_img_in)

        rows, cols = channel.shape
        crow, ccol = rows / 2, cols / 2
        f_shift[crow - 20:crow + 20, ccol - 20:ccol + 20] = 0
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        res.append(img_back)

    img_out = cv2.merge(res)  # High pass filter result

    return True, img_out


def deconvolution(img_in):

    # Write deconvolution codes here
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    # for channel in split_img:
    imf = ft(img_in, (img_in.shape[0],
                      img_in.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img_in.shape[0],
                  img_in.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf

    img_out = (ift(imconvf) * 255).astype('uint8')  # Deconvolution result

    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    # input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread("blurred2.exr",
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2LPF.png"
    output_name2 = sys.argv[4] + "2HPF.png"
    output_name3 = sys.argv[4] + "2deconv.png"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here
    level = 6
    res = []

    img_in1 = img_in1[:, :img_in1.shape[0]]
    img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

    GA, GB = img_in1, img_in2
    gpA, gpB = [GA], [GB]
    for i in range(level):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        gpA.append(GA)
        gpB.append(GB)

    lpA = [gpA[level - 1]]
    for i in range(1, level)[::-1]:
        gpa_shape = gpA[i - 1].shape
        GE = cv2.pyrUp(gpA[i], (gpa_shape[1], gpa_shape[0], gpa_shape[2]))
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    lpB = [gpB[level - 1]]
    for i in range(1, level)[::-1]:
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, level):
        ls_ = cv2.pyrUp(ls_)
        ls_ = np.add(ls_, LS[i])

    res.append(ls_)

    img_out = cv2.merge(res)  # Blending result

    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "3.png"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
