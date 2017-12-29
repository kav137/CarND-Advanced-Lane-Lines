#%%
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
#%%
class Calibrator:
    processing_log = []

    # processing related shared variables
    objpoints = []
    imgpoints = []
    pattern_size = None

    # undistortion related shared variables
    image_size = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    updated_camera_matrix = None
    roi = None

    @staticmethod
    def get_object_points():
        objp = np.zeros((Calibrator.pattern_size[0] * Calibrator.pattern_size[1], 3), dtype=np.float32)
        objp[:,:2] = np.mgrid[0:Calibrator.pattern_size[0], 0:Calibrator.pattern_size[1]].T.reshape(-1, 2)
        return objp

    @staticmethod
    def print_log():
        for line in Calibrator.processing_log:
            print(line)

    @staticmethod
    def process_calibration_image(image_path, debug=False):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_image, Calibrator.pattern_size)

        if ret:
            Calibrator.objpoints.append(Calibrator.get_object_points())
            Calibrator.imgpoints.append(corners)
            Calibrator.processing_log.append('{} - corners found successfully'.format(image_path))
        else:
            Calibrator.processing_log.append('{} - failed to find corners'.format(image_path))

        if debug:
            image_augmented = cv2.drawChessboardCorners(image, Calibrator.pattern_size, corners, ret)
            plt.title(image_path)
            plt.imshow(image_augmented)
            plt.show()

    @staticmethod
    def calculate_distortion_parameters():
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            Calibrator.objpoints,
            Calibrator.imgpoints,
            Calibrator.image_size,
            None,
            None
        )
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            dist,
            Calibrator.image_size,
            1,
            Calibrator.image_size
        )

        Calibrator.mtx = mtx
        Calibrator.dist = dist
        Calibrator.rvecs = rvecs
        Calibrator.tvecs = tvecs
        Calibrator.updated_camera_matrix = new_matrix
        Calibrator.roi = roi

    @staticmethod
    def undistort_image(image, debug=False):
        undistorted_image = cv2.undistort(
            image,
            Calibrator.mtx,
            Calibrator.dist,
            newCameraMatrix=Calibrator.updated_camera_matrix
        )

        x, y, w, h = Calibrator.roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        if debug:
            plt.figure(figsize=(10,10))
            plt.title('Original vs undistorted')
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(undistorted_image)
            plt.show()

        return undistorted_image


Calibrator.pattern_size = (9,6)
Calibrator.image_size = (1280, 720)

for image_path in glob.glob('.\\camera_cal\\*'):
    Calibrator.process_calibration_image(image_path)
Calibrator.calculate_distortion_parameters()
for image_path in glob.glob('.\\camera_cal\\*'):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Calibrator.undistort_image(image, debug=True)
