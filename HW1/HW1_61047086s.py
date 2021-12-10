import cv2
import numpy as np
# Importing the Libraries
image_file = r"C:\Users\Rohit\Desktop\Fall 2021 NTNU\Computer Vision NTU\Chapter-1\Homework\lena.bmp" # Image path: Please Copy Paste the image path here
lena = cv2.imread(image_file,cv2.COLOR_BGR2RGB) # Read Image

########################### Part 1 ##########################
########################### Using Basic Mathematics for Image transformation##################

# Blank Canvas
flipped_lena_horizontal = np.full((512, 512, 3),0, dtype = np.uint8) 
flipped_lena_vertical = np.full((512, 512, 3),0, dtype = np.uint8)
flipped_lena_diagonal = np.full((512, 512, 3),0, dtype = np.uint8)

# Dimension Values
height = lena.shape[0]
width = lena.shape[1]

# iterating over each pixel and flipping the matrix horizontally,vertically,diagonally
for h_pixel in range(height):
        for w_pixel in range(width):
            # Flip the image horizontally from co-ordinate(x,y) to co-ordinate (height-x-1,y)
            flipped_lena_horizontal[h_pixel,w_pixel] = lena[height-h_pixel-1,w_pixel]
            # Flip the image verticaally from co-ordinate(x,y) to co-ordinate (x,width-y-1)
            flipped_lena_vertical[h_pixel,w_pixel] = lena[h_pixel,width-w_pixel-1]
            # Flip the image diagonally from co-ordinate(x,y) to co-ordinate (height-x-1,width-y-1)
            flipped_lena_diagonal[h_pixel,w_pixel] = lena[height-h_pixel-1,width-w_pixel-1]


########################### Part 2 ##########################
########################### Using Functions##################

########################### Flip Image ######################
rot_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1) # Applying the transformation matrix of 45° using getRotationmatrix2D
rotated_lena = cv2.warpAffine(lena, rot_matrix, (width, height)) # Applying affine transformation so that the parallel lines in the original image will remain parallel in the output image as well

######################## Shrink Image #######################
scale_percent = 50
reduced_width = int(lena.shape[1]*scale_percent/100) # Reducing the width by 50 %
reduced_height = int(lena.shape[0]*scale_percent/100)# Reducing the height by 50 %
reduced_dimension = (reduced_height, reduced_width)
shrink_lena = cv2.resize(lena,reduced_dimension,interpolation=None) # Applying resize funtion on the lena.bmp with reduced dimension
th,binarize_lena = cv2.threshold(lena,128,255,cv2.THRESH_BINARY) # Binarizing the image using cv2.THRESH_BINARY with threshold value 128 and maximum value 255

################################# Show Image #################

cv2.imshow('Original Image',lena)
cv2.waitKey(0)
cv2.imshow('Flipped Image Horizontal',flipped_lena_horizontal)
cv2.waitKey(0)
cv2.imshow('Flipped Image Vertical',flipped_lena_vertical)
cv2.waitKey(0)
cv2.imshow('Flipped Image Diagonal',flipped_lena_diagonal)
cv2.waitKey(0)
cv2.imshow('Rotated Image',rotated_lena)
cv2.waitKey(0)
cv2.imshow('Shrinked Image',shrink_lena)
cv2.waitKey(0)
cv2.imshow('Binarized Image',binarize_lena)
cv2.waitKey(0)