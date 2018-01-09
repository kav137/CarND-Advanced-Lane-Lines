
import numpy as np
import cv2

class LineFinder:
    windows_per_frame = 9
    margin = 100
    recenter_threshold = 70
    sliding_window_height = None
    frame_height = None

    # Main function of class. Is used from the outside
    @staticmethod
    def find_lines(binary_warped, left_line=None, right_line=None):
        if (left_line is None) or (right_line is None):
            return LineFinder.find_initial_lines(binary_warped)
        else:
            return LineFinder.find_next_lines(binary_warped, left_line, right_line)

    @staticmethod
    def find_initial_lines(binary_warped):
        LineFinder.frame_height = binary_warped.shape[0]
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(LineFinder.frame_height / 2):, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        LineFinder.sliding_window_height = np.int(LineFinder.frame_height/LineFinder.windows_per_frame)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Create a list of rects in order to draw them during augmentation step
        rects = []

        # Step through the windows one by one
        for window in range(LineFinder.windows_per_frame):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = LineFinder.frame_height - (window + 1) * LineFinder.sliding_window_height
            win_y_high = LineFinder.frame_height - window * LineFinder.sliding_window_height
            win_xleft_low = leftx_current - LineFinder.margin
            win_xleft_high = leftx_current + LineFinder.margin
            win_xright_low = rightx_current - LineFinder.margin
            win_xright_high = rightx_current + LineFinder.margin

            # Remember current rect position
            rects.append([win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high])

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &
                (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &
                (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > LineFinder.recenter_threshold pixels, recenter next window on their mean position
            if len(good_left_inds) > LineFinder.recenter_threshold:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > LineFinder.recenter_threshold:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        fit_data = LineFinder.get_fit_data(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return LineFinder.apply_rects(fit_data, rects)

    @staticmethod
    def find_next_lines(binary_warped, left_line, right_line):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (nonzerox > (left_line[0]*(nonzeroy**2) + left_line[1]*nonzeroy + left_line[2] - LineFinder.margin)) &
            (nonzerox < (left_line[0]*(nonzeroy**2) + left_line[1]*nonzeroy + left_line[2] + LineFinder.margin))
        )

        right_lane_inds = (
            (nonzerox > (right_line[0]*(nonzeroy**2) + right_line[1]*nonzeroy + right_line[2] - LineFinder.margin)) &
            (nonzerox < (right_line[0]*(nonzeroy**2) + right_line[1]*nonzeroy + right_line[2] + LineFinder.margin))
        )

        fit_data = LineFinder.get_fit_data(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)
        return LineFinder.apply_polygon(fit_data)

    @staticmethod
    def get_fit_data(warped_image, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_line = np.polyfit(lefty, leftx, 2)
        right_line = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, LineFinder.frame_height - 1, LineFinder.frame_height)
        left_fitx = left_line[0] * ploty ** 2 + left_line[1] * ploty + left_line[2]
        right_fitx = right_line[0] * ploty ** 2 + right_line[1] * ploty + right_line[2]

        out_img = np.dstack((warped_image, warped_image, warped_image)) * 255
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return (left_line, right_line, left_fitx, right_fitx, ploty, out_img)


    @staticmethod
    def apply_polygon(fit_data):
        (left_line, right_line, left_fitx, right_fitx, ploty, src_img) = fit_data
        polygon_image = np.zeros_like(src_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - LineFinder.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + LineFinder.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - LineFinder.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + LineFinder.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(polygon_image, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(polygon_image, np.int_([right_line_pts]), (0, 255, 0))

        result = cv2.addWeighted(src_img, 1, polygon_image, 0.3, 0)

        return (left_line, right_line, left_fitx, right_fitx, ploty, result)

    @staticmethod
    def apply_rects(fit_data, rects):
        (left_line, right_line, left_fitx, right_fitx, ploty, src_img) = fit_data

        for rect in rects:
            win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high = rect
            # Draw the windows on the visualization image
            cv2.rectangle(src_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(src_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        return (left_line, right_line, left_fitx, right_fitx, ploty, src_img)
