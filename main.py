import cv2 as cv
import imageProcessing as ip
import classify as cf
import fourierDescriptor as fd
import imgGather as ig
import numpy as np


cap = cv.VideoCapture(0)
width = 640
height = 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

if __name__ == '__main__':
    while 1:
        flag, frame = cap.read()
        img_res, feature = ig.gather(frame)
        fd_test = np.zeros((1, 15))
        f = feature[1]
        for i in range(1, 15):
            fd_test[0, i - 1] = int(100 * feature[i] / f)

        test_knn, test_svm = cf.test_fd(fd_test)
        print("test_knn =", test_knn)
        print("test_svm =", test_svm)

        cv.imshow('frame', img_res)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("结束识别")
            break

