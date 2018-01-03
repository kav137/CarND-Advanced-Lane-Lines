#%%
import os
import glob
import time
import cv2
from src.calibrator import Calibrator
from src.edges import get_mask
#%%
Calibrator.calibrate()
test_images = glob.glob('.\\test_images\\*.jpg')
#%%
uid = str(time.asctime().replace(':', '-'))
folder = '.\\test_images\\output_{}'.format(uid)
os.mkdir(folder)

for image_path in test_images:
    image = cv2.imread(image_path)

    undistorted_image = Calibrator.undistort_image(image)
    mask = get_mask(undistorted_image)

    cv2.imwrite(image_path.replace('test_images', 'test_images\\output_{}'.format(uid)), mask)
