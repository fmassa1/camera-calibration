
import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def take_pictures():
    cap = cv2.VideoCapture() #camera info goes here
    num = 0
    while cap.isOpened():

        succes, img = cap.read()
        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite('images/img' + str(num) + '.png', img)
            print("image saved!")
            num += 1
        cv2.imshow('Img',img)
    # Release and destroy all windows before termination
    cap.release()
    cv2.destroyAllWindows()

# Apply camera calibration operation for images in the given directory path
def calibrate(square_size, width=9, height=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('images/img*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Finds the corners of the chess board
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]
# Saves the camera matrix and the distortion coefficients to given file    
def save_coefficients(mtx, dist, path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()

# Loads camera matrix and distortion coefficients from files    
def load_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    take_pictures()
    args = parser.parse_args()
    ret, mtx, dist, rvecs, tvecs = calibrate(args.square_size, args.width, args.height)
    save_coefficients(mtx, dist, args.save_file)
    print("Calibration is finished. RMS: ", ret)