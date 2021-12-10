from PIL import Image
import cv2
import numpy as np

# Importing the Libraries
image_file = r"F:\Fall 2021 NTNU\Computer Vision NTU\Chapter-4\HomeWork\lena.bmp"

gray_image = cv2.imread(image_file,0) # Read Image in Grayscale

# Dimension Values
row, col = gray_image.shape


########################### Part 1 ##########################
########################### Binarize image ##################

def img_binarize(img_in):
    image_binary = np.copy(img_in)
    for h_row in range(0, row):
        for w_col in range(0, col):
            if img_in[h_row,w_col] > 127:
                image_binary[h_row, w_col] = 255
            else:
                image_binary[h_row, w_col] = 0
    return (image_binary)

########################### Part 2 ##########################
########################### Downsample Image using  8*8 blocks ##################

# using 8*8 blocks as a unit and take the topmost-left pixel as the downsampled data
def downsampling(binary_img):
    img_down = np.zeros((64, 64), np.int)
    for h_row in range(0,img_down.shape[0]):
        for w_col in range(1,img_down.shape[1]):
            img_down[h_row][w_col] = binary_img[8 * h_row][8 * w_col]
    return img_down

########################### Part 3 ##########################
########################### Setup the Four Connectivity hFunction and fFunction ##################


# using 4 connectivity algorithm
def four_connectivity_hFunction(b,c,d,e):
    # Find the connection pattern
    if b == c and (d != b or e!= b):
        return 'q'
    if b == c and (d == b and e == b):
        return 'r'
    if b!= c:
        return 's'

def four_connectivity_fFunction(a1,a2,a3,a4):
    # Label the relation accordingly
    if ([a1, a2, a3, a4].count('r') == 4):
        # Return label 5 (interior).
        return 5
    else:
        # Return count of 'q'.
        # 0: Isolated, 1: Edge, 2: Connecting, 3: Branching, 4: Crossing.
        return [a1, a2, a3, a4].count('q')


########################### Part 4 ##########################
########################### Setup the Yokoi Connectivity Number ##################
# compute Yokoi Connectivity Number ...
def yokoi_connectivity_number(downsamples_image):
# Create a blank canvas of size 64*64 for sketching the Yokoi Number
    YokoiConnectivityNumber = np.full(downsamples_image.size, ' ')
    # Scan each column in original image.
    for h_row in range(downsamples_image.size[0]):
        # Scan each row in original image.
        for w_col in range(downsamples_image.size[1]):
            if (downsamples_image.getpixel((h_row,w_col)) != 0):
                # Get neighborhood pixel values.
                neighborhoodPixels = getNeighborhoodPixels(downsamples_image, h_row,w_col)
                # Calculating the pattern of relation between the neighboring pixel
                YokoiConnectivityNumber[h_row,w_col] = four_connectivity_fFunction(
                    four_connectivity_hFunction(neighborhoodPixels[0], neighborhoodPixels[1],
                                                neighborhoodPixels[6], neighborhoodPixels[2]),
                    four_connectivity_hFunction(neighborhoodPixels[0], neighborhoodPixels[2],
                                                neighborhoodPixels[7], neighborhoodPixels[3]),
                    four_connectivity_hFunction(neighborhoodPixels[0], neighborhoodPixels[3],
                                                neighborhoodPixels[8], neighborhoodPixels[4]),
                    four_connectivity_hFunction(neighborhoodPixels[0], neighborhoodPixels[4],
                                                neighborhoodPixels[5], neighborhoodPixels[1]))
            # This point is background.
            else:
                YokoiConnectivityNumber[h_row,w_col] = ' '

    # Return Yokoi Connectivity Number.
    return YokoiConnectivityNumber

########################### Part 5 ##########################
########################### Setup the neighboring pixel and get value function ##################
def getNeighborhoodPixels(downImage, h_row,w_col):
    """
    Corners neighborhood (for corresponding ith values in x)
    x7,x2,x6
    x3,x0,x1
    x8,x4,x5
    """
    return [getValue(downImage, h_row, w_col), getValue(downImage, h_row + 1, w_col), getValue(downImage, h_row, w_col - 1),
            getValue(downImage, h_row - 1, w_col), getValue(downImage, h_row, w_col + 1), getValue(downImage, h_row + 1, w_col + 1),
            getValue(downImage, h_row + 1, w_col - 1), getValue(downImage, h_row - 1, w_col - 1), getValue(downImage, h_row - 1, w_col + 1)]

def getValue(img, row, col):
    # Helper function to get the value separately for better code readability
    if row >= img.size[0] or row < 0 or col >= img.size[1] or col < 0:
        return 0
    return img.getpixel((row,col))

########################### Part 6 ##########################
########################### Main Function ##################
if __name__ == "__main__":
    # Converted the image to Binary in OpenCV for better readability of Code
    bin_image = img_binarize(gray_image)
    downsampled_cv_img = downsampling(bin_image)
    # Converted the OpenCV image to Pillow image for less datatype conversion constraints and better handling og RG
    downsampled_pil_img = Image.fromarray(downsampled_cv_img)
    # Get Yokoi Connectivity Number.
    YokoiConnectivityNumber = yokoi_connectivity_number(downsampled_pil_img)
    # Save Yokoi Connectivity Number to file.
    np.savetxt('YokoiConnectivityNumber.txt',YokoiConnectivityNumber.T,delimiter='', fmt='%s')
