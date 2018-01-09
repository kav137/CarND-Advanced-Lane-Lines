import numpy as np

class Measurements:
    frame_width = 1280
    meters_per_pixel_y = 30 / 720 # meters per pixel in y dimension
    meters_per_pixel_x = 3.7 / 700 # meters per pixel in x dimension

    @staticmethod
    def get_curvature(ploty, left_fitx, right_fitx):
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(
            ploty * Measurements.meters_per_pixel_y,
            left_fitx * Measurements.meters_per_pixel_x,
            2
        )
        right_fit_cr = np.polyfit(
            ploty * Measurements.meters_per_pixel_y,
            right_fitx * Measurements.meters_per_pixel_x,
            2
        )

        # Calculate the new radii of curvature
        left_curverad = (
            ((1 + (2 * left_fit_cr[0] * y_eval * Measurements.meters_per_pixel_y + left_fit_cr[1]) ** 2) **1.5) /
            np.absolute(2 * left_fit_cr[0])
        )

        right_curverad = (
            ((1 + (2 * right_fit_cr[0] * y_eval * Measurements.meters_per_pixel_y + right_fit_cr[1]) ** 2) ** 1.5) /
            np.absolute(2*right_fit_cr[0])
        )

        return (left_curverad, right_curverad)

    @staticmethod
    def get_shift(left_line, right_line):
        rightx_int = right_line[0] * 720 ** 2 + right_line[1] * 720 + right_line[2]
        leftx_int = left_line[0] * 720 ** 2 + left_line[1] * 720 + left_line[2]

        position = (rightx_int + leftx_int) / 2
        distance_from_center = ((Measurements.frame_width / 2) - position) * Measurements.meters_per_pixel_x

        return distance_from_center
