import numpy as np
import cv2 as cv

def save_camera_matrix(mtx, filename="camera_matrix.npy"):
    np.save(filename, mtx)

def load_camera_matrix(filename="camera_matrix.npy"):
    return np.load(filename)

def calibrate_camera(chessboard_images):
    objp = np.zeros((7*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    objpoints, imgpoints = [], []

    for file in chessboard_images:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 7), None)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)

    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx
    return None
