import cv2
import numpy as np
from src.birdView import BirdView

def format_text(curvature_left, curvature_right, shift):
    curvature_avg = (curvature_left + curvature_right) / 2
    shift_position = 'right' if shift > 0 else 'left'

    lines = [
        'Left line curvature: {} meters'.format(int(curvature_left)),
        'Right line curvature: {} meters'.format(int(curvature_right)),
        'Approximate road curvature: {} meters'.format(int(curvature_avg)),
        'Car is shifted {:.2f} meters to the {}'.format(abs(shift), shift_position)
    ]

    return lines

def create_road_projection(image, binary_warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, BirdView.reverted_matrix, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

def show_info(image, binary_warped, left_fitx, right_fitx, ploty, curvature_left, curvature_right, shift):
    image = create_road_projection(image, binary_warped, left_fitx, right_fitx, ploty)

    text_lines = format_text(curvature_left, curvature_right, shift)
    for index in range(len(text_lines)):
        info_text = text_lines[index]
        cv2.putText(image, info_text, (50, 50 + 35 * index), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255))

    return image
