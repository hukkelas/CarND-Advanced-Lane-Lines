import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y
CALIBRATION_BASE_PATH = "camera_cal/*.jpg"

def get_calibration_points():
    calibration_image_files = glob.glob(CALIBRATION_BASE_PATH)
    # First image is the test image
    calibration_image_files = [x for x in calibration_image_files if not x.endswith("calibration1.jpg")]
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    imgpoints = []
    objpoints = []
    # Find corners in each image
    for filepath in calibration_image_files:
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(np.copy(objp))
    return imgpoints, objpoints

def compute_calibration_coefficients():
    img = cv2.imread("camera_cal/calibration1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgpoints, objpoints = get_calibration_points()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return dist, mtx

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def rgb(img):
    return np.fliplr(img.reshape(-1,3)).reshape(img.shape)


if __name__ == '__main__':
    test_image = cv2.imread("camera_cal/calibration1.jpg")
    dist, mtx = compute_calibration_coefficients()
    undistorted = undistort(test_image, mtx, dist)
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(test_image)
    plt.subplot(1,2,2)
    plt.imshow(undistorted)
    plt.savefig("output_images/undistorted_example1.jpg")
    #plt.show()



    # On a test image
    plt.figure(figsize=(20,20))
    test_image = cv2.imread("test_images/straight_lines2.jpg")
    undistorted1 = undistort(test_image, mtx, dist)
    plt.subplot(1,2,1)
    plt.imshow(rgb(test_image))
    plt.subplot(1,2,2)
    plt.imshow(rgb(undistorted1))

    plt.savefig("output_images/undistorted_example2.jpg")
    #plt.show()