import cv2 as cv
import imageProcessing as ip
import fourierDescriptor as fd


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


if __name__ == '__main__':
    cnt_up = 0
    cnt_down = 0
    cnt_left = 0
    cnt_right = 0
    temp_cnt = 0

    cap = cv.VideoCapture(0)
    width = 640
    height = 480
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    path = './' + 'dataset' + '/'
    img_path = '/' + 'img' + '/'
    feature_path = '/' + 'feature' + '/'

    while 1:

        flag, frame = cap.read()
        res, f = gather(frame)

        cv.imshow("frame", frame)
        cv.imshow("res", res)

        direction = ''
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出数据集收集")
            break
        else:
            if key == ord('w'):
                direction = 'up'
                cnt_up += 1
                temp_cnt = cnt_up
            elif key == ord('s'):
                direction = 'down'
                cnt_down += 1
                temp_cnt = cnt_down
            elif key == ord('a'):
                direction = 'left'
                cnt_left += 1
                temp_cnt = cnt_left
            elif key == ord('d'):
                direction = 'right'
                cnt_right += 1
                temp_cnt = cnt_right

            while direction != '':
                cv.imwrite(path + direction + img_path + direction + '_' + str(temp_cnt) + '.png', res)
                print("第 " + str(temp_cnt) + " 张 " + direction + " 图片 保存成功")
                with open(path + direction + feature_path + direction + '_' + str(temp_cnt) + '.txt', 'w', encoding='utf-8') as w:
                    temp = f[1]
                    for j in range(1, len(f)):
                        x_record = int(100 * f[j] / temp)
                        w.write(str(x_record))
                        w.write(' ')
                    w.write('\n')
                print("傅里叶特征保存成功")
                direction = ''
