from PIL import Image
import numpy as np
import cv2
import math

def grayscale_Dilation(image,Kernel):
    # Convert image opencv image to pillow
    pil_image = Image.fromarray(image)
    # Copy the Image Dimension
    dilated_Image = Image.new('L', pil_image.size)
    # Centre value of the Kernel
    kernel_centre_point = tuple(kernel_items // 2 for kernel_items in Kernel.shape)
    # Iterating over each pixel in original image
    for h_row in range(pil_image.size[0]):
        for w_col in range(pil_image.size[1]):
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
                        if n_row_val < pil_image.size[0] and n_col_val < pil_image.size[1]:
                            # Get pixel value from original image at (n_row_val, n_col_val).
                            originalPixel = pil_image.getpixel((n_row_val, n_col_val))
                            # Update local max. pixel value.
                            localMaxPixel = max(localMaxPixel, originalPixel)
            # Paste minimum local pixel value on original image.
            dilated_Image.putpixel((h_row, w_col), localMaxPixel)
    # convert pillow image back to opencv image
    cv_image = np.asarray(dilated_Image)
    # Return dilated image.
    return cv_image

def grayscale_Opening(image,Kernel):
    return grayscale_Dilation(grayscale_Erosion(image,Kernel),Kernel)

def grayscale_Closing(image,Kernel):
    return grayscale_Erosion(grayscale_Dilation(image,Kernel),Kernel)

def grayscale_Erosion(image,Kernel):
    # Convert image opencv image to pillow
    pil_image = Image.fromarray(image)
    # Copy the Image Dimension
    eroded_Image = Image.new('L', pil_image.size)
    # Centre value of the Kernel
    kernel_centre_point = tuple(x // 2 for x in Kernel.shape)
    # Iterating over each pixel in original image
    for h_row in range(0,pil_image.size[0]):
        for w_col in range(0,pil_image.size[1]):
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
                        if n_row_val < pil_image.size[0] and n_col_val < pil_image.size[1]:
                            # Get pixel value from original image at (n_row_val, n_col_val).
                            originalPixel = pil_image.getpixel((n_row_val, n_col_val))
                            # Update local max. pixel value.
                            localMinPixel = min(localMinPixel, originalPixel)
            # Paste minimum local pixel value on original image.
            eroded_Image.putpixel((h_row, w_col), localMinPixel)
    # convert pillow image back to opencv image
    cv_image = np.asarray(eroded_Image)
    # Return eroded image
    return cv_image

def generate_Gaussian_noise(img, mu, sigma, amplitude):
    # This function generates Gaussian noise with the specified
    # mu, sigma, and amplitude on the given image.
    gaussianNoiseImage = np.copy(img)
    # Scan each column in original image.
    for h_row in range(gaussianNoiseImage.shape[0]):
        # Scan each row in original image.
        for w_col in range(gaussianNoiseImage.shape[1]):
            # Get pixel value with gaussian noise.
            # random.gauss function is used to return random floating point number
            noisePixel = int(gaussianNoiseImage[h_row,w_col] + amplitude * np.random.normal(mu,sigma))
            # Limit pixel value at 255.
            if noisePixel > 255:
                noisePixel = 255
            # Put pixel to noise image.
            gaussianNoiseImage[h_row,w_col] = noisePixel
    return gaussianNoiseImage

def generate_salt_and_pepper_noise(img, probability):
    # Generates salt-and-pepper noise

    img_sp = np.copy(img)
    for h_row in range(img.shape[0]):
        for w_col in range(img.shape[1]):
            randomValue = np.random.uniform(0, 1)
            if randomValue <= probability:
                img_sp[h_row][w_col] = 0
            elif randomValue >= 1 - probability:
                img_sp[h_row][w_col] = 255
            else:
                img_sp[h_row,w_col] = img[h_row,w_col]
    return img_sp

def box_filter(img, filter_size):
    # The function runs box filter with the specified filter_size on the given image.
    img_box = np.zeros(
        shape=(img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1),
        dtype=np.int)
    for h_row in range(img_box.shape[0]):
        for w_col in range(img_box.shape[1]):
            img_box[h_row][w_col] = np.mean(img[h_row: h_row + filter_size, w_col: w_col + filter_size])
    img_box = img_box.astype(np.uint8)
    return img_box

def median_filter(img, filter_size):
    # The function median filter with the specified filter_size on the given image.
    img_fil = np.zeros(
        shape=(img.shape[0] - filter_size + 1, img.shape[1] - filter_size + 1),
        dtype=np.int
    )
    for h_row in range(img_fil.shape[0]):
        for w_col in range(img_fil.shape[1]):
            img_fil[h_row][w_col] = np.median(img[h_row: h_row + filter_size, w_col: w_col + filter_size])
    img_fil = img_fil.astype(np.uint8)
    return img_fil


def getSNR(img1, img2):
    signalImage = Image.fromarray(img1)
    noiseImage = Image.fromarray(img2)

    # Clear mu and power of signal and noise.
    muSignal = 0
    powerSignal = 0
    muNoise = 0
    powerNoise = 0

    # Scan each column in signal image.
    for col in range(signalImage.size[0]):
        # Scan each row in signal image.
        for row in range(signalImage.size[1]):
            muSignal = muSignal + signalImage.getpixel((col, row))
    # Average mu of signal.
    muSignal = muSignal / (signalImage.size[0] * signalImage.size[1])

    # Scan each column in noise image.
    for col in range(noiseImage.size[0]):
        # Scan each row in noise image.
        for row in range(noiseImage.size[1]):
            muNoise = muNoise + (noiseImage.getpixel((c, r)) - signalImage.getpixel((col, row)))
    # Average mu of noise.
    muNoise = muNoise / (noiseImage.size[0] * noiseImage.size[1])

    # Scan each column in signal image.
    for col in range(signalImage.size[0]):
        # Scan each row in signal image.
        for row in range(signalImage.size[1]):
            powerSignal = powerSignal + math.pow(signalImage.getpixel((col, row)) - muSignal, 2)
    # Average power of signal.
    powerSignal = powerSignal / (signalImage.size[0] * signalImage.size[1])

    # Scan each column in noise image.
    for col in range(noiseImage.size[0]):
        # Scan each row in noise image.
        for row in range(noiseImage.size[1]):
            powerNoise = powerNoise +  math.pow((noiseImage.getpixel((col, row)) - signalImage.getpixel((col, row))) - muNoise, 2)
    # Average mu of noise.
    powerNoise = powerNoise / (noiseImage.size[0] * noiseImage.size[1])

    return 20 * math.log(math.sqrt(powerSignal) / math.sqrt(powerNoise), 10)

if __name__ == '__main__':
    # Reading the image in grayscale
    img_path = r"F:\Fall 2021 NTNU\Computer Vision NTU\Chapter-7\HomeWork-8\lena.bmp"
    img = cv2.imread(img_path, 0)
    # Generate an image with additive white Gaussian noise with amplitude = 10
    lena_gauss_10 = generate_Gaussian_noise(img, 0, 1, 10)
    # cv2.imwrite('lena_gaussian_10.bmp', lena_gauss_10)

    # Generate an image with additive white Gaussian noise with amplitude = 30
    lena_gauss_30 = generate_Gaussian_noise(img, 0, 1, 30)
    # cv2.imwrite('lena_gaussian_30.bmp', lena_gauss_30)

    # Generate an output an image with salt-and-pepper noise with probability = 0.05
    lena_sp_05 = generate_salt_and_pepper_noise(img, 0.05)
    # cv2.imwrite('lena_salt_and_pepper_0.05.bmp', lena_sp_05)


    # Generate and output an image with salt-and-pepper noise with probability = 0.1
    lena_sp_01 = generate_salt_and_pepper_noise(img, 0.10)
    # cv2.imwrite('lena_salt_and_pepper_0.1.bmp', lena_sp_01)


    # Run 3x3 box filter on the image with white Gaussian noise with amplitude = 10
    lena_gauss_10_box_3 = box_filter(lena_gauss_10, 3)
    # cv2.imwrite('lena_gaussian10_box_filter_0.3.bmp', lena_gauss_10_box_3)


    # Run 3x3 box filter on the image with white Gaussian noise with amplitude = 30
    lena_gauss_30_box_3 = box_filter(lena_gauss_30, 3)
    # cv2.imwrite('lena_gaussian30_box_filter_0.3.bmp', lena_gauss_30_box_3)


    # Run 3x3 box filter on the image with salt-and-pepper noise with threshold = 0.05
    lena_sp_05_box_3 = box_filter(lena_sp_05, 3)
    # cv2.imwrite('lena_sp05_box_filter_0.3.bmp', lena_sp_05_box_3)


    # Run 3x3 box filter on the image with salt-and-pepper noise with threshold = 0.1
    lena_sp_01_box_3 = box_filter(lena_sp_01, 3)
    # cv2.imwrite('lena_sp01_box_filter_0.3.bmp', lena_sp_01_box_3)


    # Run 5x5 box filter on the image with white Gaussian noise with amplitude = 10
    lena_gauss_10_box_5 = box_filter(lena_gauss_10, 5)
    # cv2.imwrite('lena_gaussian10_box_filter_0.5.bmp', lena_gauss_10_box_5)


    # Run 5x5 box filter on the image with white Gaussian noise with amplitude = 30
    lena_gauss_30_box_5 = box_filter(lena_gauss_30, 5)
    # cv2.imwrite('lena_gaussian30_box_filter_0.5.bmp', lena_gauss_30_box_5)


    # Run 5x5 box filter on the image with salt-and-pepper noise with threshold = 0.05
    lena_sp_05_box_5 = box_filter(lena_sp_05, 5)
    # cv2.imwrite('lena_sp05_box_filter_0.5.bmp', lena_sp_05_box_5)


    # Run 5x5 box filter on the image with salt-and-pepper noise with threshold = 0.1
    lena_sp_01_box_5 = box_filter(lena_sp_01, 5)
    # cv2.imwrite('lena_sp01_box_filter_0.5.bmp', lena_sp_01_box_5)


    # Run 3x3 median filter on the image with white Gaussian noise with amplitude = 10
    lena_gauss_10_med_3 = median_filter(lena_gauss_10, 3)
    # cv2.imwrite('lena_gauss_10_median_filter_3.bmp', lena_gauss_10_med_3)


    # Run 3x3 median filter on the image with white Gaussian noise with amplitude = 30
    lena_gauss_30_med_3 = median_filter(lena_gauss_30, 3)
    # cv2.imwrite('lena_gauss_30_median_filter_3.bmp', lena_gauss_30_med_3)


    # Run 3x3 median filter on the image with salt-and-pepper noise with threshold = 0.05
    lena_sp_05_med_3 = median_filter(lena_sp_05, 3)
    # cv2.imwrite('lena_sp_05_median_filter_3.bmp', lena_sp_05_med_3)


    # Run 3x3 median filter on the image with salt-and-pepper noise with threshold = 0.1
    lena_sp_01_med_3 = median_filter(lena_sp_01, 3)
    # cv2.imwrite('lena_sp_10_median_filter_3.bmp', lena_sp_01_med_3)


    # Run 5x5 median filter on the image with white Gaussian noise with amplitude = 10
    lena_gauss_10_med_5 = median_filter(lena_gauss_10, 5)
    # cv2.imwrite('lena_gauss_10_median_filter_5.bmp', lena_gauss_10_med_5)


    # Run 5x5 median filter on the image with white Gaussian noise with amplitude = 30
    lena_gauss_30_med_5 = median_filter(lena_gauss_30, 5)
    # cv2.imwrite('lena_gauss_30_median_filter_5.bmp', lena_gauss_30_med_5)


    # Run 5x5 median filter on the image with salt-and-pepper noise with threshold = 0.05
    lena_sp_05_med_5 = median_filter(lena_sp_05, 5)
    # cv2.imwrite('lena_sp_05_median_filter_5.bmp', lena_sp_05_med_5)


    # Run 5x5 median filter on the image with salt-and-pepper noise with threshold = 0.1
    lena_sp_01_med_5 = median_filter(lena_sp_01, 5)
    # cv2.imwrite('lena_sp_10_median_filter_5.bmp', lena_sp_01_med_5)


    # Use octagon as kernel and set the orgin is at the center
    kernel = np.array([
        [-2, -1], [-2, 0], [-2, 1],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        [2, -1], [2, 0], [2, 1]
    ])

    # closing followed by opening on all noisy images
    lena_gauss_10_close_open = grayscale_Opening(grayscale_Closing(lena_gauss_10, kernel), kernel)
    lena_gauss_30_close_open = grayscale_Opening(grayscale_Closing(lena_gauss_30, kernel), kernel)
    lena_sp_05_close_open = grayscale_Opening(grayscale_Closing(lena_sp_05, kernel), kernel)
    lena_sp_01_close_open = grayscale_Opening(grayscale_Closing(lena_sp_01, kernel), kernel)

    # cv2.imwrite('lena_gaussian_10_close_open.bmp', lena_gauss_10_close_open)

    # cv2.imwrite('lena_gaussian_30_close_open.bmp', lena_gauss_30_close_open)

    # cv2.imwrite('lena_sp_05_close_ope.bmp', lena_sp_05_close_open)

    # cv2.imwrite('lena_sp_01_close_open.bmp', lena_sp_01_close_open)


    # opening followed by closing on all noisy images
    lena_gauss_10_open_close = grayscale_Closing(grayscale_Opening(lena_gauss_10, kernel), kernel)
    lena_gauss_30_open_close = grayscale_Closing(grayscale_Opening(lena_gauss_30, kernel), kernel)
    lena_sp_05_open_close = grayscale_Closing(grayscale_Opening(lena_sp_05, kernel), kernel)
    lena_sp_01_open_close = grayscale_Closing(grayscale_Opening(lena_sp_01, kernel), kernel)

    # cv2.imwrite('lena_gaussian_10_open_close.bmp', lena_gauss_10_open_close)

    # cv2.imwrite('lena_gaussian_30_open_close.bmp', lena_gauss_30_open_close)

    # cv2.imwrite('lena_sp_05_open_close.bmp', lena_sp_05_open_close)

    # cv2.imwrite('lena_sp_01_open_close.bmp', lena_sp_01_open_close)

    # Saving the SNR in a txt file
    # Calculate SNR for all noise image.
    gaussianNoise_10_SNR = getSNR(img, lena_gauss_10)
    gaussianNoise_30_SNR = getSNR(img, lena_gauss_30)
    saltAndPepper_0_01_SNR = getSNR(img, lena_sp_01)
    saltAndPepper_0_05_SNR = getSNR(img, lena_sp_05)

    gaussianNoise_10_box_3x3_SNR = getSNR(img, lena_gauss_10_box_3)
    gaussianNoise_30_box_3x3_SNR = getSNR(img, lena_gauss_30_box_3)
    saltAndPepper_0_01_box_3x3_SNR = getSNR(img, lena_sp_01_box_3)
    saltAndPepper_0_05_box_3x3_SNR = getSNR(img, lena_sp_05_box_3)
    gaussianNoise_10_box_5x5_SNR = getSNR(img, lena_gauss_10_box_5)
    gaussianNoise_30_box_5x5_SNR = getSNR(img, lena_gauss_10_box_5)
    saltAndPepper_0_01_box_5x5_SNR = getSNR(img, lena_sp_01_box_5)
    saltAndPepper_0_05_box_5x5_SNR = getSNR(img, lena_sp_05_box_5)

    gaussianNoise_10_median_3x3_SNR = getSNR(img, lena_gauss_10_med_3)
    gaussianNoise_30_median_3x3_SNR = getSNR(img, lena_gauss_30_med_3)
    saltAndPepper_0_01_median_3x3_SNR = getSNR(img, lena_sp_01_med_3)
    saltAndPepper_0_05_median_3x3_SNR = getSNR(img, lena_sp_05_med_3)
    gaussianNoise_10_median_5x5_SNR = getSNR(img, lena_gauss_10_med_5)
    gaussianNoise_30_median_5x5_SNR = getSNR(img, lena_gauss_30_med_5)
    saltAndPepper_0_01_median_5x5_SNR = getSNR(img, lena_sp_01_med_5)
    saltAndPepper_0_05_median_5x5_SNR = getSNR(img, lena_sp_05_med_5)

    gaussianNoise_10_openingThenClosing_SNR = getSNR(img, lena_gauss_10_open_close)
    gaussianNoise_30_openingThenClosing_SNR = getSNR(img, lena_gauss_30_open_close)
    saltAndPepper_0_01_openingThenClosing_SNR = getSNR(img, lena_sp_01_open_close)
    saltAndPepper_0_05_openingThenClosing_SNR = getSNR(img, lena_sp_05_open_close)

    gaussianNoise_10_closingThenOpening_SNR = getSNR(img, lena_gauss_10_close_open)
    gaussianNoise_30_closingThenOpening_SNR = getSNR(img, lena_gauss_30_close_open)
    saltAndPepper_0_01_closingThenOpening_SNR = getSNR(img, lena_sp_01_close_open)
    saltAndPepper_0_05_closingThenOpening_SNR = getSNR(img, lena_sp_05_close_open)

    # Write SNR to text file.
    file = open("SNR.txt", "w")
    file.write("gaussianNoise_10_SNR: " + str(gaussianNoise_10_SNR) + '\n')
    file.write("gaussianNoise_30_SNR: " + str(gaussianNoise_30_SNR) + '\n')
    file.write("saltAndPepper_0_01_SNR: " + str(saltAndPepper_0_01_SNR) + '\n')
    file.write("saltAndPepper_0_05_SNR: " + str(saltAndPepper_0_05_SNR) + '\n')

    file.write("gaussianNoise_10_box_3x3_SNR: " + str(gaussianNoise_10_box_3x3_SNR) + '\n')
    file.write("gaussianNoise_30_box_3x3_SNR: " + str(gaussianNoise_30_box_3x3_SNR) + '\n')
    file.write("saltAndPepper_0_01_box_3x3_SNR: " + str(saltAndPepper_0_01_box_3x3_SNR) + '\n')
    file.write("saltAndPepper_0_05_box_3x3_SNR: " + str(saltAndPepper_0_05_box_3x3_SNR) + '\n')
    file.write("gaussianNoise_10_box_5x5_SNR: " + str(gaussianNoise_10_box_5x5_SNR) + '\n')
    file.write("gaussianNoise_30_box_5x5_SNR: " + str(gaussianNoise_30_box_5x5_SNR) + '\n')
    file.write("saltAndPepper_0_01_box_5x5_SNR: " + str(saltAndPepper_0_01_box_5x5_SNR) + '\n')
    file.write("saltAndPepper_0_05_box_5x5_SNR: " + str(saltAndPepper_0_05_box_5x5_SNR) + '\n')

    file.write("gaussianNoise_10_median_3x3_SNR: " + str(gaussianNoise_10_median_3x3_SNR) + '\n')
    file.write("gaussianNoise_30_median_3x3_SNR: " + str(gaussianNoise_30_median_3x3_SNR) + '\n')
    file.write("saltAndPepper_0_01_median_3x3_SNR: " + str(saltAndPepper_0_01_median_3x3_SNR) + '\n')
    file.write("saltAndPepper_0_05_median_3x3_SNR: " + str(saltAndPepper_0_05_median_3x3_SNR) + '\n')
    file.write("gaussianNoise_10_median_5x5_SNR: " + str(gaussianNoise_10_median_5x5_SNR) + '\n')
    file.write("gaussianNoise_30_median_5x5_SNR: " + str(gaussianNoise_30_median_5x5_SNR) + '\n')
    file.write("saltAndPepper_0_01_median_5x5_SNR: " + str(saltAndPepper_0_01_median_5x5_SNR) + '\n')
    file.write("saltAndPepper_0_05_median_5x5_SNR: " + str(saltAndPepper_0_05_median_5x5_SNR) + '\n')

    file.write("gaussianNoise_10_openingThenClosing_SNR: " + str(gaussianNoise_10_openingThenClosing_SNR) + '\n')
    file.write("gaussianNoise_30_openingThenClosing_SNR: " + str(gaussianNoise_30_openingThenClosing_SNR) + '\n')
    file.write("saltAndPepper_0_01_openingThenClosing_SNR: " + str(saltAndPepper_0_01_openingThenClosing_SNR) + '\n')
    file.write("saltAndPepper_0_05_openingThenClosing_SNR: " + str(saltAndPepper_0_05_openingThenClosing_SNR) + '\n')

    file.write("gaussianNoise_10_closingThenOpening_SNR: " + str(gaussianNoise_10_closingThenOpening_SNR) + '\n')
    file.write("gaussianNoise_30_closingThenOpening_SNR: " + str(gaussianNoise_30_closingThenOpening_SNR) + '\n')
    file.write("saltAndPepper_0_01_closingThenOpening_SNR: " + str(saltAndPepper_0_01_closingThenOpening_SNR) + '\n')
    file.write("saltAndPepper_0_05_closingThenOpening_SNR: " + str(saltAndPepper_0_05_closingThenOpening_SNR) + '\n')