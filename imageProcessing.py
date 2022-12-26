import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import fourierDescriptor as fd


###############################################
# 图像截取函数
# 输入：原图，截取起始点的横坐标，纵坐标，截取的宽度、长度
# 输出：截取后的图像， 被标记截取范围后的原图
###############################################
def crop(img_no_crop, x0, y0, width, height):
    img_no_crop_rectangle = img_no_crop.copy()
    cv.rectangle(img_no_crop_rectangle, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))
    img_crop = img_no_crop[x0:x0 + width, y0:y0 + height]

    return img_crop, img_no_crop_rectangle


###############################################
# 椭圆皮肤检测函数
# 输入：原图
# 输出：筛选后只剩下皮肤的图
# 原理：在YCrCb空间，肤色像素点会聚集到一个椭圆区域。先定义一个椭圆模型，然后将每个RGB像素点转换到YCrCb空间比对是否在椭圆区域，是的话判断为皮肤。
###############################################
def skin_ellipse(img_crop):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)  # 绘制椭圆弧线
    YCrCb = cv.cvtColor(img_crop, cv.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, Cr, Cb) = cv.split(YCrCb)  # 拆分出Y,Cr,Cb值
    skin = np.zeros(Cr.shape, dtype=np.uint8)  # 掩膜
    (x, y) = Cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if skinCrCbHist[Cr[i][j], Cb[i][j]] > 0:  # 若不在椭圆区间中
                skin[i][j] = 255
    res = cv.bitwise_and(img_crop, img_crop, mask=skin)
    return res


###############################################
# YCrCb+阈值分割图像检测函数
# 输入：原图
# 输出：筛选后只剩下皮肤的图
# 原理:针对YCrCb中Cr分量的处理，对CR通道单独进行Otsu处理，Otsu方法opencv里用threshold，Otsu算法是对图像的灰度级进行聚类。
###############################################
def skin_threshold(roi):
    YCrCb = cv.cvtColor(roi, cv.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Ostu处理
    res = cv.bitwise_and(roi, roi, mask=skin)
    return res


###############################################
# 形态学处理函数
# 输入：提取过皮肤以后的图像（存在瑕疵的干扰）
# 输出：排除大部分瑕疵后的图像
###############################################
def morphology(img_skin):
    # k = np.ones((16, 16), np.uint8)
    k = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), None)
    img_erode = cv.erode(img_skin, k, 1)
    img_dilate = cv.dilate(img_erode, k, 1)
    img_erode = cv.erode(img_dilate, k, 1)
    img_dilate = cv.dilate(img_erode, k, 1)

    return img_dilate


# 逐个图片显示
# if __name__ == '__main__':
#     path = 'img/1.jpg'
#     img = cv.imread(path)
#     img_1, img_2 = crop(img, 150, 400, 500, 500)
#     img_1 = cv.GaussianBlur(img_1, (3, 3), 0)
#     img_3 = skin_ellipse(img_1)
#     img_4 = morphology(img_3)
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.imshow(img_2[:, :, ::-1])
#     plt.subplot(2, 2, 2)
#     plt.imshow(img_1[:, :, ::-1])
#     plt.subplot(2, 2, 3)
#     plt.imshow(img_3[:, :, ::-1])
#     plt.subplot(2, 2, 4)
#     plt.imshow(img_4[:, :, ::-1])
#     plt.show()

# 对摄像头进行设置
cap = cv.VideoCapture(0)
width = 640
height = 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

if __name__ == '__main__':

    while 1:
        # 读取摄像头画面
        flag, frame = cap.read()
        # 对摄像头画面进行截取
        img, img_origin = crop(frame, 0, 0, width, height)
        # 对图像进行高斯模糊
        img = cv.GaussianBlur(img, (3, 3), 0)
        # 进行皮肤提取
        img_ellipse = skin_ellipse(img)

        img_morphology = morphology(img_ellipse)

        img_fourier, _ = fd.fourierDesciptor(img_morphology)

        cv.imshow("ellipse", img_ellipse)
        cv.imshow("morphology", img_morphology)
        cv.imshow("fourier", img_fourier)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
