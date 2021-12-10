import cv2
import numpy as np

# Importing the Libraries
image_file = r"F:\Fall 2021 NTNU\Computer Vision NTU\Chapter-7\HomeWork-10\lena.bmp"
gray_image = cv2.imread(image_file, 0)  # Read Image in Grayscale


def convolve(img, kernel):
    row_img, col_img = img.shape
    row_k, col_k = kernel.shape
    res = 0
    for h_row in range(row_img):
        for w_col in range(col_img):
            if 0 <= row_img - h_row - 1 < row_k and 0 <= col_img - w_col - 1 < col_k:
                res += img[h_row, w_col] * kernel[row_img - h_row - 1, col_img - w_col - 1]
    return res


def convolution_image(gray_image,kernel, threshold):
    # To avoid index out of bounds Error
    mask_img = np.zeros((gray_image.shape[0] - kernel.shape[0] + 1, gray_image.shape[1] - kernel.shape[1] + 1))
    for h_row in range(mask_img.shape[0]):
        for w_col in range(mask_img.shape[1]):
            value = convolve(gray_image[h_row:h_row + kernel.shape[0], w_col:w_col + kernel.shape[1]], kernel)
            if value >= threshold:
                mask_img[h_row, w_col] = 1
            elif value <= -threshold:
                mask_img[h_row, w_col] = -1
            else:
                mask_img[h_row, w_col] = 255

    return mask_img


def zero_crossing(gray_image):
    # 8 neighborhood zero crossing
    zero_cross_image = np.zeros(gray_image.shape)
    for h_row in range(0, gray_image.shape[0] - 1):
        for w_col in range(0, gray_image.shape[1] - 1):
            if gray_image[h_row][w_col] > 0:
                if gray_image[h_row + 1][w_col] < 0 or gray_image[h_row + 1][w_col + 1] < 0 or gray_image[h_row][w_col + 1] < 0:
                    zero_cross_image[h_row, w_col] = 1
            elif gray_image[h_row][w_col] < 0:
                if gray_image[h_row + 1][w_col] > 0 or gray_image[h_row + 1][w_col + 1] > 0 or gray_image[h_row][w_col + 1] > 0:
                    zero_cross_image[h_row, w_col] = 1
    return zero_cross_image
if __name__ == "__main__":
    # Laplace mask 1
    laplace_mask1_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplace_mask1_image = convolution_image(gray_image,laplace_mask1_kernel, 15)
    cv2.imshow('Laplace Mask 1 Image', laplace_mask1_image)
    cv2.waitKey(0)
    # Laplace Mask 2
    laplace_mask2_kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])/3
    laplace_mask2_image = convolution_image(gray_image, laplace_mask2_kernel, 15)
    cv2.imshow('Laplace Mask 2 Image', laplace_mask2_image)
    cv2.waitKey(0)
    # Minimum Variance Laplacian
    minimum_variance_laplacian_kernel = np.array([
            [2., -1, 2],
            [-1, -4, -1],
            [2, -1, 2]
        ]) / 3
    minimum_variance_laplacian_image = convolution_image(gray_image, minimum_variance_laplacian_kernel, 15)
    cv2.imshow('Minimum Variance Laplacian Image', minimum_variance_laplacian_image)
    cv2.waitKey(0)
    # Laplace of Gaussian
    laplace_of_gaussian_kernel = np.array([
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
    ])
    laplace_of_gaussian_image = convolution_image(gray_image, laplace_of_gaussian_kernel, 3000)
    cv2.imshow('Laplace of Gaussian Image', laplace_of_gaussian_image)
    cv2.waitKey(0)
    # Difference of Gaussian
    difference_of_gaussian_kernel = np.array([
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
    ])
    difference_of_gaussian_image = convolution_image(gray_image, difference_of_gaussian_kernel, 1)
    cv2.imshow('Difference of Gaussian Image', difference_of_gaussian_image)
    cv2.waitKey(0)
    zero_crossing_image = zero_crossing(difference_of_gaussian_image)
    cv2.imshow('Zero Crossing Image', zero_crossing_image)
    cv2.waitKey(0)




