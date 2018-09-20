import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def sobel_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    sobelx = np.uint8(255*sobelx/np.max(sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(sobelx)
    sxbinary[(sobelx >= thresh_min) & (sobelx <= thresh_max)] = 1
    return sxbinary


def hls_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary

def compute_binary_image(img):
    sobel = sobel_threshold(img)
    hls_binary = hls_threshold(img)
    combined_binary = np.zeros_like(sobel)
    combined_binary[(sobel == 1) | (hls_binary == 1)] = 1
    return combined_binary
    


if __name__ == '__main__':

    test_image_files = glob.glob('test_images/*.jpg')
    for f in test_image_files:
        img = cv2.imread(f)
        plt.figure(figsize=(20,20))
        sobel = sobel_threshold(img)
        plt.imshow(sobel, cmap="gray")
        hls_binary = hls_threshold(img)
        color_binary = np.dstack(( np.zeros_like(sobel), sobel, hls_binary)) * 255
        plt.subplot(2,2,3)
        plt.imshow(color_binary)
        combined_binary = np.zeros_like(sobel)
        combined_binary[(sobel == 1) | (hls_binary == 1)] = 1
        plt.subplot(2,2,1)
        
        plt.imshow(combined_binary, cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(img)
        plt.show()

