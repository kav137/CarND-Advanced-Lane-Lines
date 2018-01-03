#%%
import cv2
import math
import cv2
import numpy as np
#%%
def get_sobel(image, direction='x', threshold=(50, 100), kernel_size=9, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel = None
    if direction == 'x':
        sobel = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif direction == 'y':
        sobel = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)


    sobel_abs = np.absolute(sobel)
    sobel_max = np.max(sobel)

    sobel_normalized = np.uint8(255 * sobel_abs / sobel_max)
    mask = np.zeros_like(sobel_normalized)
    mask[(sobel_normalized >= threshold[0]) & (sobel_normalized <= threshold[1])] = 255 if debug else 1

    if debug:
        cv2.imwrite('./results/sobel/{}_k-{}_{}.png'.format(direction, threshold, kernel_size), mask)

    return mask

def get_sobel_magnitude(image, threshold=(50, 100), kernel_size=9, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))

    sobelxy_max = np.max(sobelxy)
    sobelxy_normalized = np.uint8(sobelxy * 255 / sobelxy_max)

    mask = np.zeros_like(sobelxy)
    mask[(sobelxy_normalized >= threshold[0]) & (sobelxy_normalized <= threshold[1])] = 255 if debug else 1

    if debug:
        cv2.imwrite('./results/sobelxy/xy_k-{}_{}.png'.format(kernel_size, threshold), np.uint8(mask))

    return mask

def get_sobel_angular(image, threshold=(0.8, 1.2), kernel_size=9, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_ang = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    mask = np.zeros_like(sobel_ang)
    mask[(sobel_ang >= threshold[0]) & (sobel_ang <= threshold[1])] = 255 if debug else 1

    if debug:
        cv2.imwrite('./results/angular/ang_k-{}_{}.png'.format(kernel_size, threshold), np.uint8(mask))

    return mask

def get_hls_mask(image, threshold_s=(150, 230), threshold_h=(15, 35)):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    s_channel = hls_image[:,:,2]
    h_channel = hls_image[:,:,0]
    mask = np.zeros_like(s_channel)

    mask[
        (s_channel >= threshold_s[0]) & (s_channel <= threshold_s[1]) &
        (h_channel >= threshold_h[0]) & (h_channel <= threshold_h[1])
    ] = 1

    return mask

def get_mask(image, debug=False):
    sobelx = get_sobel(image, 'x', (50, 150), 15)
    sobely = get_sobel(image, 'y', (50, 150), 15)
    sobelxy = get_sobel_magnitude(image, (50, 150), 9)
    angular = get_sobel_angular(image)
    hls = get_hls_mask(image)

    if debug:
        image_new = np.dstack((angular, hls, sobelxy)) * 255
        return image_new

    combined_mask = np.zeros_like(sobelx)
    # combined_mask[((sobelx == 1) & (sobely == 1)) & ((sobelxy == 1) | (angular == 1) | (hls == 1))] = 255
    combined_mask[((sobelx == 1) & (sobely == 1) & (angular == 1)) | ((sobelxy == 1) | (hls == 1))] = 255


    return combined_mask
