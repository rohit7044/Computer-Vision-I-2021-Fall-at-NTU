import cv2
import numpy as np
import matplotlib.pyplot as plt
# Importing the Libraries
image_file = r"C:\Users\Rohit\Desktop\Fall 2021 NTNU\Computer Vision NTU\Chapter-2\Homework\lena.bmp"
original_Image = cv2.imread(image_file)
image = cv2.imread(image_file,0) # Read Image in Grayscale

# Dimension Values
row = image.shape[0]
col = image.shape[1]

########################### Part 1 ##########################
########################### Binarize image ##################

def img_binarize(img_in):
    image_pixel_check = (img_in > 0x7f) * 0xff
    image_binary = (image_pixel_check == 0xff) * 1
    return (image_binary)

########################### Part 2 ##########################
########################### Show Histogram ##################
def show_histogram(image):
    image_histogram = np.zeros([256], np.int32)
    for h_row in range(0, row):
        for w_col in range(0, col):
            image_histogram[image[h_row, w_col]] += 1
    # Creating histogram
    plt.plot(image_histogram)
    plt.title("Histogram of Original Image")
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.show()

########################### Part 3 ##########################
######################## Connected Components #######################

###### disjoint set union and find algorithm ###
def union_find(label):
    original_label = label
    cnt = 0
    row,col = image.shape
    global op_cnt
    # Check if already labelled
    while label != parent_label[label] and cnt < row * col:
        op_cnt += 1
        label = parent_label[label]
        cnt += 1
    return label


############### draw the result rectangle ######
def draw_rect(up, down, left, right):
    cv2.rectangle(image, (left, up), (right, down), (0,0,0), 1)

############### draw the result centroid #######
def draw_cent(cen_i, cen_j):
    SHIFT = 10
    cv2.line(image, (cen_j - SHIFT, cen_i), (cen_j + SHIFT, cen_i), (255,255,255), 2)
    cv2.line(image, (cen_j, cen_i - SHIFT), (cen_j, cen_i + SHIFT), (255,255,255), 2)

############### CC main function ###############
op_cnt = 0
parent_label = []
cc_img = img_binarize(image)


def connected_components():
    global op_cnt
    LABEL_THRESHOLD = 500
    # set parent label
    row, col = cc_img.shape
    for pixels in range(row * col):
        parent_label.append(pixels)

    # do connected components using 4 connected neighbours (left and up)
    label = 2
    for h_row in range(row):
        for w_col in range(col):
            ok1 = 0
            ok2 = 0
            op_cnt += 1
            if cc_img[h_row, w_col] == 1:
                if w_col - 1 >= 0 and cc_img[h_row, w_col - 1] > 1:  # left has already labeled
                    cc_img[h_row, w_col] = union_find(cc_img[h_row, w_col - 1])
                    ok1 = 1

                if h_row - 1 >= 0 and cc_img[h_row - 1, w_col] > 1:  # up has already labeled
                    if ok1:  # set the connected component to make left = up as the same group
                        parent_label[cc_img[h_row, w_col]] = union_find(cc_img[h_row - 1, w_col])
                    else:
                        cc_img[h_row, w_col] = cc_img[h_row - 1, w_col]

                    ok2 = 1

                if ok2 == 0 and ok1 == 0:
                    cc_img[h_row, w_col] = label
                    label += 1

    # union and find merging
    for h_row in range(row):
        for w_col in range(col):
            op_cnt += 1
            if cc_img[h_row, w_col] > 1:
                cc_img[h_row, w_col] = union_find(cc_img[h_row, w_col])

    # statistical data for label threshold > 500
    mymap = [0 for pixel in range(row * col)]
    for h_row in range(0, row):
        for w_col in range(0, col):
            mymap[cc_img[h_row,w_col]] += 1
    cc_pos = {}
    cc_value = []
    for h_row in range(0, row):
        for w_col in range(0, col):
            if cc_img[h_row,w_col] and cc_img[h_row,w_col] not in cc_value and mymap[cc_img[h_row,w_col]] > LABEL_THRESHOLD:
                cc_value.append(cc_img[h_row,w_col])
    for i in cc_value:
        cc_pos[i] = []
    for h_row in range(0, row):
        for w_col in range(0, col):
            if cc_img[h_row,w_col] and mymap[cc_img[h_row, w_col]] > LABEL_THRESHOLD:
                cc_pos[cc_img[h_row, w_col]].append((h_row, w_col))

    # draw the rectangles and centroid

    for each_cc_value in cc_value:
        up = min(cc_pos[each_cc_value], key=lambda u: u[0])[0]
        down = max(cc_pos[each_cc_value], key=lambda d: d[0])[0]
        left = min(cc_pos[each_cc_value], key=lambda l: l[1])[1]
        right = max(cc_pos[each_cc_value], key=lambda r: r[1])[1]
    # Centroid = average of positions of connected component pixels
        cen_i, cen_j = [sum(i) / len(i) for i in zip(*cc_pos[each_cc_value])]
        cen_i = int(cen_i)
        cen_j = int(cen_j)

        draw_rect(up, down, left, right)
        draw_cent(cen_i, cen_j)


if __name__ == "__main__":
    connected_components()
    # Show Original image
    cv2.imshow('Original Image',original_Image)
    # Show histogram
    show_histogram(image)
    # Show Binarized image
    bin_image = img_binarize(image)
    plt.title("Binarized Image")
    plt.imshow(bin_image, cmap = 'gray')
    plt.show()
    # Show labelled Image
    cv2.imshow('Labeled Image',image)
    cv2.waitKey(0)