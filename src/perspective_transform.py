import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from camera_calibration import compute_calibration_coefficients, undistort
from thresholding import compute_binary_image

def get_src_points(img):
    height, width = img.shape[:2]
    src = np.float32([
        (width*310//700, height*19//30),
        (width*390//700, height*19//30),
        (width*83//90, height),
        (width*10//90, height),
        (width*31//70, height*19//30),
    ])
    return src
def get_dst_points(img):
    height, width = img.shape[:2]    
    dst = np.float32([
        (width*10//90, 0),
        (width*83//90, 0),
        (width*83//90, height),
        (width*10//90, height),
        (width*10//90, 0),        
    ])
    return dst
def perspective_projection(gray):
    # (x, y)
    src = get_src_points(gray)
    dst = get_dst_points(gray)

    
    M = cv2.getPerspectiveTransform(src[:-1], dst[:-1])
    img_size =(gray.shape[1], gray.shape[0])
    warped = cv2.warpPerspective(gray, M, img_size)
    return warped, M

if __name__ == '__main__':
    test_image_files = glob.glob('test_images/*.jpg')
    dist, mtx = compute_calibration_coefficients()
    for f in test_image_files:

        plt.figure(figsize=(20,20))
        img = cv2.imread(f)
        plt.subplot(2,2,1)
        plt.imshow(img)
        img = undistort(img, mtx, dist)
        plt.subplot(2,2,2)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,2,3)
        gray = compute_binary_image(img)
        plt.imshow(gray, cmap='gray')        
        warped, M = perspective_projection(gray)
        plt.subplot(2,2,4)
        plt.imshow(warped, cmap="gray")
        plt.show()



