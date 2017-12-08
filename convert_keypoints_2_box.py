import glob
import numpy as np
import json
from pprint import pprint
import cv2
import imutils


files = glob.glob("/home/xc/dataset/mpii/keypoint/*.json")
files = np.sort(files)

def keypoints_2_box(keys, wrist):
    xs = [wrist[0]]
    ys = [wrist[1]]
    # xs = []
    # ys = []
    for i in xrange(len(keys)/3):
        if keys[i*3+2] < 0.1:
            continue
        xs.append(keys[i*3])
        ys.append(keys[i*3+1])
    xs = filter(lambda x: x != 0, xs)
    if len(xs) == 0:
        xs = [0]
    ys = filter(lambda x: x != 0, ys)
    if len(ys) == 0:
        ys = [0]
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def area(box):
    ar = (box[2]-box[0])*(box[3]-box[1])
    return ar

def box_2_string(box, img_w, img_h):
    bb = np.array(box)
    bb = bb.astype(np.float)
    bb = [(bb[0]+bb[2])/(img_w*2.), (bb[1]+bb[3])/(2*img_h), (bb[2]-bb[0])/img_w, (bb[3]-bb[1])/img_h]
    result = '1' + " " + " ".join([str(a) for a in bb]) + '\n'
    return result

# for i in range(len(files)):
for i in range(2500):
    print i
    data = json.load(open(files[i]))

    img_file = files[i].replace('_keypoints.json', '.jpg')
    img_file = img_file.replace('keypoint','images')
    label_path = img_file.replace('images','labels')
    label_path = label_path.replace('.jpg','.txt')
    out_file = open(label_path, 'w')
    img = cv2.imread(img_file)

    img_w = np.shape(img)[1]
    img_h = np.shape(img)[0]
    for pp in data['people']: # each peopel has its own pose
        hl_box = keypoints_2_box(pp['hand_left_keypoints'], pp['pose_keypoints'][21:23])
        if area(hl_box)<100:
            hl_box = [0,0,0,0]
        hr_box = keypoints_2_box(pp['hand_right_keypoints'], pp['pose_keypoints'][12:14])
        if area(hr_box)<100:
            hr_box = [0,0,0,0]
        if hl_box != [0,0,0,0]:
            # cv2.rectangle(img, (hl_box[0],hl_box[1]), (hl_box[2],hl_box[3]), (255,0,0), 3)
            out_file.write(box_2_string(hl_box, img_w, img_h))

        if hr_box != [0,0,0,0]:
            # cv2.rectangle(img, (hr_box[0],hr_box[1]), (hr_box[2],hr_box[3]), (0,255,0), 3)
            out_file.write(box_2_string(hr_box, img_w, img_h))

    
    # img = imutils.resize(img, width=640)
    # cv2.imshow('box', img)
    # cv2.waitKey(0)