import cv2
import glob
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from camera_calibration import compute_calibration_coefficients, undistort, rgb
from thresholding import compute_binary_image
from perspective_transform import perspective_projection


def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:, :]
    
    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

def find_start_position(img):
    # Take a histogram of the bottom half of the image
    histogram = hist(img)
    # Create an output image to draw on and visualize the result
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    
    midpoint = histogram.shape[0]//2
    leftx = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint
    return leftx, rightx


def find_lane_pixels(img):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    leftx_current, rightx_current = find_start_position(img)
    out_img = np.dstack((img, img, img)) * 255
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzeroy, nonzerox = img.nonzero()

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)    

    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr

def fit_poly(binary_warped, leftx, lefty, rightx, righty):

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_x_coordinate(left_fit, ploty)
    right_fitx = get_x_coordinate(right_fit, ploty)
    return None, left_fit, right_fit, ploty, left_fitx, right_fitx

def get_x_coordinate(fit, ploty):
    return fit[0]*ploty**2 + fit[1]*ploty + fit[2]

# using a previous found polynomial to try to fit a new one onto the warped image. 
def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    out_img, left_fit, right_fit, ploty, left_fitx, right_fitx = fit_poly(binary_warped, leftx, lefty, rightx, righty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)    
    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr

# Assume a lane is 3.7m
def distance_to_center(img, left_fit, right_fit):
    
    leftx = left_fit[-1]
    rightx = right_fit[-1]
    xm_per_pix = 3.7 / (rightx - leftx) 
    center = (leftx + (rightx - leftx) / 2) * xm_per_pix
    position = (img.shape[1] * xm_per_pix) / 2
    distance = abs(center - position)
    return distance


class Lines:

    def __init__(self):
        self.history_size = 10
        self.left_fit, self.right_fit = None, None
        self.reset_count = 0
        self.left_fits = []
        self.right_fits = []
    
    def find_lines(self, img, warped, Minv):
        if self.left_fit is None or self.right_fit is None:
            # No lane lines found before
            out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = fit_polynomial(warped)
        else:
            out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = search_around_poly(warped, self.left_fit, self.right_fit)

        
              
        if self.should_reset(ploty, left_fit, right_fit, left_fitx, right_fitx, left_fit_cr, right_fit_cr):
            self.reset_count += 1
            if self.left_fit is not None and self.right_fit is not None:
                left_fitx = get_x_coordinate(self.left_fit, ploty)
                right_fitx = get_x_coordinate(self.right_fit, ploty)
            if self.reset_count > 10:
                print("Reset fit")
                self.left_fit, self.right_fit = None, None
                self.right_fits = []
                self.left_fits = []

        else:
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)                  
            self.left_fit = np.mean(self.left_fits, axis=0)
            self.right_fit = np.mean(self.right_fits, axis=0)
            left_fitx = get_x_coordinate(self.left_fit, ploty)
            right_fitx = get_x_coordinate(self.right_fit, ploty)
            if len(self.left_fits) > self.history_size:
                del self.left_fits[0]
                del self.right_fits[0]
            self.reset_count = 0
            distance = distance_to_center(img, self.left_fit, self.right_fit)
        return map_lane_lines_to_original(img, warped, left_fitx, right_fitx, ploty, Minv)
    
    def distance_to_center(self, img):
        return distance_to_center(img, self.left_fit, self.right_fit)
    # Sanity check
    def should_reset(self, ploty, left_fit, right_fit, left_fitx, right_fitx, left_fit_cr, right_fit_cr):

        # 
        left_curverad, right_curverad = measure_curvature(ploty, left_fit_cr, right_fit_cr)
        self.left_curverad, self.right_curverad = left_curverad, right_curverad
        if left_curverad*0.9 <= right_curverad and left_curverad*1.1 >= right_curverad:
            return True
        
        distance = right_fitx - left_fitx
        if (distance < 0).sum() > 0: # If the right lane is crossing left lane
            return True
        std = distance.std()
        distance -= distance.mean()
        if (np.abs(distance) > std*2.5).sum() > 0:
            return True

        

        # Should be within 10% error
        return 

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
def curvature(fit, y):
    A, B, C = fit
    top = (1 + (2*A*y*ym_per_pix + B )**2)**(3/2)
    bottom = np.abs(2*A)
    return top / bottom

def measure_curvature(ploty, left_fit_cr, right_fit_cr):

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = curvature(left_fit_cr, y_eval)  ## Implement the calculation of the left line here
    right_curverad = curvature(right_fit_cr, y_eval)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def map_lane_lines_to_original(img, warped, left_fitx, right_fitx, ploty, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


if __name__ == '__main__':
    
    test_image_files = [
        "test_images/test6.jpg",
        "test_images/test4.jpg",
        "test_images/test1.jpg",
        "test_images/test5.jpg"
    ]
    test_image_files = glob.glob('test_images/*.jpg')
    dist, mtx = compute_calibration_coefficients()
    line = Lines()
    for f in test_image_files[:1]:
        line = Lines()
        """
        plt.figure(figsize=(10,10))
        img = cv2.imread(f)

        img = undistort(img, mtx, dist)

        gray = compute_binary_image(img)
        warped, M = perspective_projection(gray)
        Minv = np.linalg.inv(M)
        histogram = hist(warped)
        plt.subplot(2,2,1)
        plt.imshow(warped, cmap="gray")
        plt.plot(warped.shape[0]-histogram)

        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        plt.subplot(2,2,2)
        plt.imshow(out_img)
  

        out_img, left_fit, right_fit, ploty, left_fitx, right_fitx = fit_polynomial(warped)
                
        plt.subplot(2,2,3)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')      
        plt.imshow(out_img)
        left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
        plt.subplot(2,2,4)
        result = map_lane_lines_to_original(img, warped, left_fitx, right_fitx, ploty, Minv)

        plt.imshow(result)
        """
        plt.figure(figsize=(10,10))
        img = rgb(cv2.imread(f))

        img = undistort(img, mtx, dist)
        
        gray = compute_binary_image(img)
        warped, M = perspective_projection(gray)
        Minv = np.linalg.inv(M)
        

        result = line.find_lines(img, warped, Minv)
        #print(line.left_curverad, line.right_curverad)
        #print(line.distance_to_center(img))
        plt.title(f"Distance to Center= {line.distance_to_center(img):.3f}, Left curverad = {line.left_curverad:.1f}, Right Curverad={line.right_curverad:.1f}")
        plt.imshow( result)        
        plt.savefig("output_images/lane_finding_output.jpg")
        #plt.show()




