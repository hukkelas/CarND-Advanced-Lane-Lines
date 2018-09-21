
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
import cv2
import glob
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from camera_calibration import compute_calibration_coefficients, undistort
from thresholding import compute_binary_image
from perspective_transform import perspective_projection
from lane_finding import measure_curvature_pixels, fit_polynomial, map_lane_lines_to_original
from moviepy.editor import VideoFileClip

dist, mtx = compute_calibration_coefficients()



def find_lines(img):
  img = undistort(img, mtx, dist)
  
  gray = compute_binary_image(img)
  warped, M = perspective_projection(gray)
  Minv = np.linalg.inv(M)

  out_img, left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped)
  left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
  result = map_lane_lines_to_original(img, warped, left_fitx, right_fitx, ploty, Minv)
  return result



def process_video(input_path, output_path="output_images/test.mp4"):
  ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
  ## To do so add .subclip(start_second,end_second) to the end of the line below
  ## Where start_second and end_second are integer values representing the start and end of the subclip
  ## You may also uncomment the following line for a subclip of the first 5 seconds
  ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
  clip1 = VideoFileClip(input_path)
  white_clip = clip1.fl_image(find_lines) #NOTE: this function expects color images!!
  white_clip.write_videofile(output_path, audio=False)


if __name__ == '__main__':
  if True:
    #process_video("project_video.mp4")
    #process_video("challenge_video.mp4", 'challenge_video_out.mp4')
    process_video("harder_challenge_video.mp4", "output_images/harder_challenge_video_out.mp4")
  else:
    test_image_files = glob.glob('test_images/*.jpg')
    dist, mtx = compute_calibration_coefficients()
    for f in test_image_files:
        plt.figure(figsize=(10,10))
        img = cv2.imread(f)
        result = find_lines(img)
        plt.imshow(result)
        plt.show()