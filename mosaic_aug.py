from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import math
from sklearn.utils import shuffle


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty, size_w, size_h):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0: # 左上角图
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
                # # 判断是否有很大的框子出现，这种基本都是错的
                # cut_w, cut_h = x2 - x1, y2-y1
                # if cut_w > ((3/4)*cutx) or cut_h > ((3/4)*cuty):
                #     continue



            if i == 1: # 左下角图
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
                # # 判断是否有很大的框子出现，这种基本都是错的
                # cut_w, cut_h = x2 - x1, y2 - y1
                # if cut_w > ((3/4)* cutx) or cut_h > ((3/4)* (size_h-cuty)):
                #     continue


            if i == 2: # 右下角图
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
                # # 判断是否有很大的框子出现，这种基本都是错的
                # cut_w, cut_h = x2 - x1, y2 - y1
                # if cut_w > ((3/4) * (size_w-cutx)) or cut_h > ((3/4) * (size_h - cuty)):
                #     continue

            if i == 3: # 右上角图
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
                # # 判断是否有很大的框子出现，这种基本都是错的
                # cut_w, cut_h = x2 - x1, y2 - y1
                # if cut_w > ((3/4) * (size_w - cutx)) or cut_h > ((3/4) * cuty):
                #     continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255

        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h

            # # 判断是否有很大的bbox，这种基本都是错的
            # box_w = box[:, 2] - box[:, 0]
            # box_h = box[:, 3] - box[:, 1]
            # box = box[np.logical_and(box_w < (1*w/2), box_h < (1*h/2))]

            # # 此处判断框子是否超出裁剪的范围，超出则删除
            # box_w = box[:, 2] - box[:, 0]
            # box_h = box[:, 3] - box[:, 1]
            # box = box[np.logical_and(box_w > 1, box_h > 1)]

            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        # # 对每个图片进行标签可视化
        # img = Image.fromarray((image_data * 255).astype(np.uint8))
        # for j in range(len(box_data)):
        #     thickness = 3
        #     left, top, right, bottom = box_data[j][0:4]
        #     draw = ImageDraw.Draw(img)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        # img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty, w, h)

    return new_image, new_boxes


def normal_(annotation_line, input_shape):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    iw, ih = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    box[:, [0, 2]] = iw - box[:, [2, 0]]

    return image, box


if __name__ == "__main__":
    # with open("2007_train.txt") as f:
    # with open("/Users/videopls/Desktop/标注已完成/own_labelme_img_yolo3.txt") as f:
    # with open("/Users/videopls/Desktop/ownimg_6coco/6coco/annotations/6coco_train.txt") as f:
    # with open("/Users/videopls/Desktop/ownimg_6coco/all_img/oriimg_train.txt", 'r') as f:
    # with open("/Users/videopls/Desktop/ownimg_6coco/own_labelme_img_yolo3.txt", 'r') as f:
    # # with open("/Users/videopls/Desktop/ownimg_6coco/all_img/add_txt/addimg_train.txt") as f:
    #     lines = f.readlines()

    with open("/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_ori+flip_yolov3.txt", 'r') as f1:
    # with open("/Users/videopls/Desktop/paosawu_all/ori_paosawu_img_yolo3.txt", 'r') as f1:
        lines = f1.readlines()
    # with open("/Users/videopls/Desktop/paosawu_all/ori_paosawu_img_yolo3_413.txt", 'r') as f2:
    #     lines2 = f2.readlines()
    # with open("/Users/videopls/Desktop/ownimg_6coco/own_images_4-9.txt", 'r') as f3:
    #     lines3 = f3.readlines()
    # with open("/Users/videopls/Desktop/train12551+json/sixcoco12251_train.txt", 'r') as f4:
    #     lines4 = f4.readlines()


    # lines = lines + lines2 + lines3 + lines4

    # w = open("/Users/videopls/Desktop/ownimg_6coco/own_addimg_train.txt", 'w')
    # w = open("/Users/videopls/Desktop/paosawu_all/add_sixcoco+ownimg+paosawu.txt", 'w')
    w = open("/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_mosaic.txt", 'w')

    # for num in range(8):
    # for num in range(8, 16):
    # for num in range(16, 24):
    # for num in range(24, 28):
    # for num in range(28, 29):
    for num in range(29, 30):
        lines = shuffle(lines, random_state=num)
        # a = np.random.randint(0, len(lines))


        # index = 0
        # line_all = lines[a:a+4]
        # for line in line_all:
        #     image_data, box_data = normal_(line,[416,416])
        #     img = image_data
        #     for j in range(len(box_data)):
        #         thickness = 3
        #         left, top, right, bottom  = box_data[j][0:4]
        #         draw = ImageDraw.Draw(img)
        #         for i in range(thickness):
        #             draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(255,255,255))
        #     img.show()
        #     # img.save(str(index)+"box.jpg")
        #     index = index+1


        a = 0
        while a <= len(lines)-4:

            line = lines[a:a + 4]
            # for i in line:
            #     print(i)
            # image_data, box_data = get_random_data(line, [416, 416])
            # image_data, box_data = get_random_data(line, [1080, 1920])
            image_data, box_data = get_random_data(line, [640, 640])
            img = Image.fromarray((image_data * 255).astype(np.uint8))

            # # 打印框子
            # for j in range(len(box_data)):
            #     thickness = 3
            #     left, top, right, bottom, cls = box_data[j]
            #     draw = ImageDraw.Draw(img)
            #     for i in range(thickness):
            #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
            #         draw.text((left + i, top + i-40), str(int(cls)), (255, 0, 0), font=ImageFont.truetype("AppleMyungjo.ttf", 40, encoding="utf-8"))


            # 保存图片
            name = "/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_mosaic/%s_%s.jpg" % (num, a)
            img.save(name)



            w.write(name)
            for j in range(len(box_data)):
                x_min, y_min, x_max, y_max, cls = box_data[j]
                box_info = " %d,%d,%d,%d,%d" % (x_min, y_min, x_max, y_max, cls)
                w.write(box_info)
            w.write("\n")

            # img.show()
            # img.save("box_all.jpg")

            a += 4

    # w.close()

