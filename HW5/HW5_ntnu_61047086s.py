# import cv2
import numpy as np
from PIL import Image

# Importing the Libraries
image_file = r"C:\Users\Rohit\Desktop\Fall 2021 NTNU\Computer Vision NTU\Chapter-5\HomeWork\61047086s_HW5_ver1\lena.bmp"
gray_image = Image.open(image_file)


# Dimension Values
row = gray_image.size[0]
col = gray_image.size[1]

# Kernel Size
Kernel = np.array([
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
])


########################### Part 1 ##########################
########################### Grayscale Image Dilation Morphology ##################


def grayscale_Dilation(image,Kernel):
    # Copy the Image Dimension
    dilated_Image = Image.new('L', image.size)
    # Centre value of the Kernel
    kernel_centre_point = tuple(x // 2 for x in Kernel.shape)
    # Iterating over each pixel in original image
    for h_row in range(row):
        for w_col in range(col):
            # Initialize local pixel value and record it
            localMaxPixel = 0
            # Iterating over Kernel Shape
            for K_row in range(Kernel.shape[0]):
                for K_col in range(Kernel.shape[1]):
                    # Get kernel value 1 from each iteration
                    if Kernel[K_row, K_col] == 1:
                        # Get the destination pixel location to be filled out
                        n_row_val = h_row + (K_row - kernel_centre_point[0])
                        n_col_val = w_col + (K_col - kernel_centre_point[1])
                        # Avoiding out of range and putting the value 255 to destination pixel
                        if n_row_val < image.size[0] and n_col_val < image.size[1]:
                            # Get pixel value from original image at (n_row_val, n_col_val).
                            originalPixel = image.getpixel((n_row_val, n_col_val))
                            # Update local max. pixel value.
                            localMaxPixel = max(localMaxPixel, originalPixel)
            # Paste minimum local pixel value on original image.
            dilated_Image.putpixel((h_row, w_col), localMaxPixel)
    # Return dilated image.
    return dilated_Image

########################### Part 2 ##########################
########################### Grayscale Image Erosion Morphology ##################


def grayscale_Erosion(image,Kernel):
    # Copy the Image Dimension
    eroded_Image = Image.new('L', image.size)
    # Centre value of the Kernel
    kernel_centre_point = tuple(x // 2 for x in Kernel.shape)
    # Iterating over each pixel in original image
    for h_row in range(0,row):
        for w_col in range(0,col):
            # Initialize local pixel value and record it
            localMinPixel = 255
            # Iterating over Kernel Shape
            for K_row in range(0, Kernel.shape[0]):
                for K_col in range(0, Kernel.shape[1]):
                    # Get kernel value 1 from each iteration
                    if Kernel[K_row, K_col] == 1:
                        # Get the destination pixel location to be filled out
                        n_row_val = h_row + (K_row - kernel_centre_point[0])
                        n_col_val = w_col + (K_col - kernel_centre_point[1])
                        # Avoiding out of range and putting the value 255 to destination pixel
                        if n_row_val < image.size[0] and n_col_val < image.size[1]:
                            # Get pixel value from original image at (n_row_val, n_col_val).
                            originalPixel = image.getpixel((n_row_val, n_col_val))
                            # Update local max. pixel value.
                            localMinPixel = min(localMinPixel, originalPixel)
            # Paste minimum local pixel value on original image.
            eroded_Image.putpixel((h_row, w_col), localMinPixel)
    # Return eroded image.
    return eroded_Image

########################### Part 3 ##########################
######################## Grayscale Image Opening Morphology #######################


def grayscale_Opening(image,Kernel):
    return grayscale_Dilation(grayscale_Erosion(image,Kernel),Kernel)

########################### Part 4 ##########################
######################## Grayscale Image Closing Morphology #######################


def grayscale_Closing(image,Kernel):
    return grayscale_Erosion(grayscale_Dilation(image,Kernel),Kernel)


if __name__ == "__main__":
    dilated_image = grayscale_Dilation(gray_image,Kernel)
    eroded_image = grayscale_Erosion(gray_image,Kernel)
    opening_image = grayscale_Opening(gray_image,Kernel)
    closing_image = grayscale_Closing(gray_image,Kernel)

    dilated_image.show()
    eroded_image.show()
    opening_image.show()
    closing_image.show()


