import cv2 as cv
import imageProcessing as ip
import fourierDescriptor as fd

cap = cv.VideoCapture(0)
width = 640
height = 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


###############################################
# 皮肤提取最终函数
# 输入：原图
# 输出：最终效果图,傅里叶特征
###############################################
def gather(img):
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    img_ellipse = ip.skin_ellipse(img_blur)
    img_morphology = ip.morphology(img_ellipse)
    img_fourier, feature = fd.fourierDesciptor(img_morphology)
    feature = abs(feature)

    return img_fourier, feature


path = './' + 'dataset' + '/'
img_path = '/' + 'img' + '/'
feature_path = '/' + 'feature' + '/'


if __name__ == '__main__':
    cnt_up = 0
    cnt_down = 0
    cnt_left = 0
    cnt_right = 0

    while 1:

        flag, frame = cap.read()
        res, f = gather(frame)

        cv.imshow("frame", frame)
        cv.imshow("res", res)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出数据集收集")
            break
        elif key == ord('w'):
            cnt_up += 1
            cv.imwrite(path + 'up' + img_path + 'up_' + str(cnt_up) + '.png', res)
            print("第  " + str(cnt_up) + "  up图片保存成功")
        elif key == ord('s'):
            cnt_down += 1
            cv.imwrite(path + 'down' + img_path + 'down_' + str(cnt_down) + '.png', res)
            print("第  " + str(cnt_down) + "  down图片保存成功")
        elif key == ord('a'):
            cnt_left += 1
            cv.imwrite(path + 'left' + img_path + 'left_' + str(cnt_left) + '.png', res)
            print("第  " + str(cnt_left) + "  left图片保存成功")
        elif key == ord('d'):
            cnt_right += 1
            cv.imwrite(path + 'right' + img_path + 'right_' + str(cnt_right) + '.png', res)
            print("第  " + str(cnt_right) + "  right图片保存成功")
