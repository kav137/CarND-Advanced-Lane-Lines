import cv2
import numpy as np

class BirdView:
    matrix = None
    reverted_matrix = None

    @staticmethod
    def transform_perspective(image):
        src_coords = np.float32([(596, 460), (713, 460), (1019, 666), (321, 666)])
        dest_coords = np.float32([(250, 0), (1030, 0), (1030, 720), (250, 720)])

        if BirdView.matrix is None:
            BirdView.matrix = cv2.getPerspectiveTransform(src_coords, dest_coords)
            BirdView.reverted_matrix = cv2.getPerspectiveTransform(dest_coords, src_coords)

        warped = cv2.warpPerspective(image, BirdView.matrix, (1280, 720), flags = cv2.INTER_LINEAR)

        return warped
