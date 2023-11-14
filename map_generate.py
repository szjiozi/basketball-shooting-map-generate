# -*- coding: utf-8 -*-

# import sys
# import time
# from models.tiny_yolo import TinyYoloNet
import cv2
from pytorch_YOLOv4.tool.utils import *
from pytorch_YOLOv4.tool.torch_utils import *
from pytorch_YOLOv4.tool.darknet2pytorch import Darknet
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import ginput
import numpy as np

"""hyper parameters"""
use_cuda = False


def getPoint(img):
    plt.figure(figsize=(18, 10))
    plt.imshow(img[:, :, ::-1])
    plt.title('Choose four points counter clockwise, first point top-left')
    points = []

    for i in range(4):
        x, y = ginput(1)[0]
        plt.scatter(x, y)
        points.append([int(x), int(y)])
    plt.close()
    print(points)
    return points


def find_corner(img, hoop_center):
    img = img.copy()
    kernel = np.ones((4, 9), np.uint8)
    while True:
        new_img = img.copy()
        b = img[:, :, 0]
        b = cv2.convertScaleAbs(b, alpha=1.2, beta=-50)
        canny = cv2.Canny(b, 15, 200)
        canny = cv2.dilate(canny, kernel)

        cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]
        exist = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, peri * 0.02, True)
            if len(approx) == 4:
                rec = approx
                exist = 1
                break
        if exist:
            cv2.drawContours(new_img, [rec], -1, (0, 0, 255), 2)
            points = [list(rec[0][0]), list(rec[1][0]), list(rec[2][0]), list(rec[3][0])]
            if points[0][0] > hoop_center:
                points = [points[3], points[0], points[1], points[2]]
            cv2.circle(new_img, (points[0][0], points[0][1]), 5, (255, 0, 0), -1)

        cv2.putText(new_img, "Press 'y' to confirm or Press 'n' to manually choose point",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(new_img, "Press 'y' to confirm or Press 'n' to manually choose point",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 240, 240), 2)
        cv2.imshow("video input", new_img)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('y') and exist:    # when press y
            cv2.destroyAllWindows()
            return points

        elif key & 0xFF == ord('n') or key & 0xFF == 27:    # when press n or esc
            cv2.destroyAllWindows()
            return getPoint(img)


def return_position(img, boxes, man_pos, ball_pos, hoop_pos, tracker):
    width = img.shape[1]
    height = img.shape[0]
    detected = [0, 0, 0]
    for i in range(len(boxes)):
        box = boxes[i]
        if box[6] == 1 and not detected[1]:
            ball_pos[0] = int(box[0] * width)
            ball_pos[1] = int(box[1] * height)
            ball_pos[2] = int(box[2] * width)
            ball_pos[3] = int(box[3] * height)
            detected[1] = 1
            tracker.init(img, (ball_pos[0], ball_pos[1], ball_pos[2] - ball_pos[0], ball_pos[3] - ball_pos[1]))
        if box[6] == 0 and not detected[0]:
            man_pos[0] = int(box[0] * width)
            man_pos[1] = int(box[1] * height)
            man_pos[2] = int(box[2] * width)
            man_pos[3] = int(box[3] * height)
            detected[0] = 1
        if box[6] == 2 and not detected[2]:
            hoop_pos[0] = int(box[0] * width)+10
            hoop_pos[1] = int(box[1] * height)+10
            hoop_pos[2] = int(box[2] * width)-10
            hoop_pos[3] = int(box[3] * height)-10
            detected[2] = 1
    """
    if not detected[1]:
        success, bbox = tracker.update(img)
        if success:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            ball_pos = [x, y, x+w, y+h]
    """
    return man_pos, ball_pos, hoop_pos, tracker


def return_hoop(img, boxes):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        if box[6] == 2:
            hoop_pos = [int(box[0] * width)+10, int(box[1] * height)+10, int(box[2] * width)-10, int(box[3] * height)-10]
            return hoop_pos
    points = getPoint(img)
    return [points[0][0], points[0][1], points[2][0], points[2][1]]


def evaluate_hoop(img, ref):
    value = np.mean(ref.astype("float")) - np.mean(img.astype("float"))
    return value


def detect_and_build_map(cfgfile, weightfile, videoPath, mapImgPath,
                         save=False, processedVideoPath=None):
    import cv2
    m = Darknet(cfgfile)
    tracker = cv2.TrackerCSRT_create()
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        m.cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videoPath)
    im_dst = cv2.imread('dataset/2DCourtBasketball.jpeg')

    # save output video as mp4
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = height/im_dst.shape[0]
    flat_width = int(ratio*im_dst.shape[1])
    im_dst = cv2.resize(im_dst, (flat_width, height))
    if save:
        process_video = cv2.VideoWriter(processedVideoPath,
                                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (width+flat_width, height))

    # read the first frame for initialize
    ret, img = cap.read()

    # initialize the position for future use
    man = [-50, -50, -10, -10]
    ball = [0, height-5, 5, height]
    tracker.init(img, (ball[0], ball[1], ball[2] - ball[0], ball[3] - ball[1]))
    shooting = 0
    ball_in_hand = 1

    # find the position of the hoop
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    hoop = return_hoop(img, boxes[0])
    hoop_w = hoop[2] - hoop[0]
    hoop_h = hoop[3] - hoop[1]
    reference = img.copy()[hoop[1]:hoop[3], hoop[0]:hoop[2]]

    # Calculate Homography
    pts_src = np.array(find_corner(img, (hoop[0]+hoop[2])/2))
    dis_x = hoop[0] - pts_src[0][0]
    dis_y = hoop[1] - pts_src[0][1]
    pts_dst = np.array([[250, 67], [430, 67], [430, 290], [250, 290]])
    h_matrix, status = cv2.findHomography(pts_src, pts_dst)

    # set args to detect for every f frames
    c = 0
    f = 10

    # for detecting whether the ball goals
    shoot_count = []
    # plt.ion()
    # plt.figure(1)
    # x_value = []
    # y_value = []
    print('loop start')
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # detect for every f frames, else use tracker
        if c % f == 0:
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))
            man, ball, hoop, tracker = return_position(img, boxes[0], man, ball, hoop, tracker)
            hoop[2] = hoop[0] + hoop_w
            hoop[3] = hoop[1] + hoop_h

        # track for ball
        success, bbox = tracker.update(img)
        if success:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            ball = [x, y, x + w, y + h]

        cv2.circle(img, (hoop[0] - dis_x, hoop[1] - dis_y), 5, (255, 0, 0), -1)
        dx = hoop[0] - dis_x - pts_src[0][0]
        dy = hoop[1] - dis_y - pts_src[0][1]
        for i in pts_src:
            i[0] += dx
            i[1] += dy

        # check situation based on position
        if not shooting and ball[3] < man[1]-(man[3]-man[1])/3 and ball_in_hand:
            h_matrix, status = cv2.findHomography(pts_src, pts_dst)
            pos = (int((man[0] + man[2]) / 2), man[3]-10)
            cv2.circle(img, pos, 5, (255, 255, 0), -1)
            pos_2d = np.dot(h_matrix, np.array([pos[0], pos[1], 1]))
            pos_2d = pos_2d / pos_2d[2]
            shoot_count.append([int(pos_2d[0]), int(pos_2d[1]), 0])
            shooting = 1
            ball_in_hand = 0
            ball_ref = img.copy()[ball[1]:ball[3], ball[0]:ball[2], 0]
            reference = img.copy()[hoop[1]:hoop[3], hoop[0]:hoop[2], 0]

        if ball[1] > man[1]+50 and shooting:
            shooting = 0

        if man[0]-50 < (ball[0]+ball[2])/2 < man[2]+50 \
                and man[1]-50 < (ball[1]+ball[3])/2 < man[3]+50:
            ball_in_hand = 1

        # see if the ball goals
        if shooting:
            """    
            plt.clf()
            x_value.append(c)
            area = img.copy()[hoop[1]:hoop[3], hoop[0]:hoop[2], 0]
            value = evaluate_hoop(area, reference, ball_ref)
            y_value.append(value)
            plt.plot(x_value, y_value, '-r')
            plt.pause(0.01)
            """
            area = img.copy()[hoop[1]:hoop[3], hoop[0]:hoop[2], 0]
            value = evaluate_hoop(area, reference)
            if value > 7.5:
                shoot_count[-1][2] = 1

        # plot the positions of objects in image
        cv2.rectangle(img, (ball[0], ball[1]), (ball[2], ball[3]), (0, 0, 255), 2)
        cv2.rectangle(img, (man[0], man[1]), (man[2], man[3]), (255, 255, 0), 2)
        cv2.rectangle(img, (hoop[0], hoop[1]), (hoop[2], hoop[3]), (255, 0, 0), 2)
        cv2.drawContours(img, [pts_src], -1, (0, 255, 0), 2)
        if shooting:
            cv2.putText(img, "Shooting", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 240), 2)
        elif ball_in_hand:
            cv2.putText(img, "Ready to shoot", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 240), 2)

        flat_img = im_dst.copy()
        goal_count = 0
        for [x, y, g] in shoot_count:
            if g:
                cv2.circle(flat_img, (x, y), 8, (255, 0, 0), -1)
                goal_count += 1
            else:
                cv2.circle(flat_img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(flat_img, "Shoot num: "+str(len(shoot_count)),
                    (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 240, 240), 2)
        cv2.putText(flat_img, "Goal num: " + str(goal_count),
                    (400, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (240, 240, 240), 2)
        whole_img = np.hstack((img, flat_img))
        resize_whole = cv2.resize(whole_img, (0, 0), None, 0.7, 0.7)
        cv2.imshow('whole', resize_whole)
        c += 1
        if save:
            process_video.write(whole_img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    if save:
        process_video.release()
    cv2.destroyAllWindows()
    cv2.imshow('2D-graph', flat_img)
    cv2.imwrite(mapImgPath, flat_img)
    if cv2.waitKey(0) == 27:
        return
    # plt.ioff()
    # plt.show()


if __name__ == '__main__':
    cfgfile = 'pytorch_YOLOv4/cfg/yolov4-basketball.cfg'
    weightfile = 'pytorch_YOLOv4/yolov4-basketball.weights'
    videoname = 'input_4.mp4'

    videoPath = "dataset/" + videoname
    mapimgpath = 'dataset/map_' + videoname[:-3] + 'jpg'
    process_video = 'dataset/detecting_' + videoname
    process_map = 'dataset/mapping_' + videoname
    detect_and_build_map(cfgfile, weightfile, videoPath, mapimgpath
                         , save=True, processedVideoPath=process_video)
