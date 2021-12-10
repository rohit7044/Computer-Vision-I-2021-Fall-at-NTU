import cv2
import numpy as np
import matplotlib.pyplot as plt
# Importing the Libraries

########################### Part 1 ##########################
########################### Original Image and it's Histogram ##################
def show_original_image():
    image_file = r"C:\Users\Rohit\Desktop\Fall 2021 NTNU\Computer Vision NTU\Chapter-2\Homework\lena.bmp"
    image = cv2.imread(image_file, 0)  # Read Image in Grayscale
    cv2.imshow("Original Image",image)
    cv2.waitKey(0)
    return image


def show_histogram(image):
    image_histogram = np.zeros([256], np.int32)
    for h_row in range(0, image.shape[0]):
        for w_col in range(0, image.shape[1]):
            image_histogram[image[h_row, w_col]] += 1
    # Creating histogram
    plt.plot(image_histogram)
    plt.title("Histogram of Image")
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.show()
    return image_histogram


########################### Part 2 ##########################
########################### Dark Image and it's Histogram ##################


def dark_image_converter(image):
    dark_image = np.copy(image)
    for h_pixel in range(image.shape[0]):
        for w_pixel in range(image.shape[1]):
            dark_image[h_pixel,w_pixel] = image[h_pixel,w_pixel]//3
    cv2.imshow("Dark Image",dark_image)
    cv2.waitKey(0)
    return dark_image


########################### Part 3 ##########################
########################### Histogram Equalization on Part 2 image ##################


def histogram_equalization_image(D_hist,D_image):
    height, width = D_image.shape
    histogram_equalized_image = np.copy(D_image)
    transformationTable = np.zeros(256)
# Following Histogram Equalization method shown in class s = T(r)
    for items in range(len(transformationTable)):
        intensity_sum = np.sum(D_hist[0:items + 1])
        transformationTable[items] = 255 * intensity_sum / height / width
    for h_row in range(0, D_image.shape[0]):
        for w_col in range(0, D_image.shape[1]):
            histogram_equalized_image[h_row, w_col] = transformationTable[D_image[h_row,w_col]]
    cv2.imshow("Equalized Image", histogram_equalized_image)
    cv2.waitKey(0)
    return histogram_equalized_image


if __name__ == "__main__":
    # Show original Image and histogram
    original_gray_image = show_original_image()
    original_histogram = show_histogram(original_gray_image)
    # Show Dark image and its histogram
    dark_image = dark_image_converter(original_gray_image)
    dark_hist = show_histogram(dark_image)
    # Histogram Equalization
    equi_image = histogram_equalization_image(dark_hist,dark_image)
    equi_histogram =show_histogram(equi_image)

