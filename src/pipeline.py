#%%
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

%matplotlib inline
#%%
from src.calibrator import Calibrator
from src.edges import get_mask
from src.birdView import BirdView
from src.lineFinder import LineFinder
from src.measurements import Measurements
from src.utils import show_info
#%%
Calibrator.calibrate()

left_line = None
right_line = None

def pipeline(image):
    undistorted_image = Calibrator.undistort_image(image)
    transformed_image = BirdView.transform_perspective(undistorted_image)
    mask = get_mask(transformed_image)

    global left_line, right_line

    (left_line, right_line, left_fitx, right_fitx, ploty, result_img) = LineFinder.find_lines(
        mask,
        left_line,
        right_line
    )

    left_curvature, right_curvature = Measurements.get_curvature(ploty, left_fitx, right_fitx)
    shift = Measurements.get_shift(left_line, right_line)

    annotated_image = show_info(
        undistorted_image, mask, left_fitx, right_fitx, ploty, left_curvature, right_curvature, shift
    )

    return annotated_image

clip_name = VideoFileClip('project_video.mp4')
output_name = 'result_video.mp4'

output = clip_name.fl_image(pipeline)
%time output.write_videofile(output_name, audio=False)
#%%

