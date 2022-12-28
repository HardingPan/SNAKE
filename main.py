import cv2 as cv
import imageProcessing as ip
import classify as cf
import fourierDescriptor as fd
import imgGather as ig
import numpy as np
import serial

# 初始化摄像头
cap = cv.VideoCapture(0)
width = 640
height = 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


if __name__ == '__main__':
    # 设置串口
    ser = serial.Serial("COM3", 9600)
    # cnt用于计数结果相同次数
    cnt = 0
    # temp用于记录上次发送的串口信号
    temp = 0
    # last_knn记录上次test_knn的指
    last_knn = [0]
    while 1:
        flag, frame = cap.read()
        img_res, feature = ig.gather(frame)
        fd_test = np.zeros((1, 15))
        f = feature[1]
        for i in range(1, 15):
            fd_test[0, i - 1] = int(100 * feature[i] / f)

        test_knn, test_svm = cf.test_fd(fd_test)
        # print("test_knn =", test_knn)
        # print("test_svm =", test_svm)
        print("knn网络识别结果为  ", test_knn)
        # print("svm网络识别结果为  ", test_svm)
        cv.imshow('frame', img_res)

        cnt = cnt + 1
        if test_knn != last_knn:
            cnt = 0
            last_knn = test_knn
        print(cnt)
        s = test_knn[0]
        if cnt > 3 & temp != s:
            temp = s
            ser.write(str(s).encode('utf-8'))
            print("发送串口数据  ", s, "\n")

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("结束识别")
            break
