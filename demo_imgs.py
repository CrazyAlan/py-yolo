from utils import *
from darknet import Darknet
import cv2
import imageio
import numpy as np
import imutils
import glob
from PIL import Image, ImageDraw


def demo(cfgfile, weightfile, folder):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    imgs = glob.glob(folder+'/*.jpg')
    imgs = np.sort(imgs)
    # cap = cv2.VideoCapture(0)
    # reader = imageio.get_reader('<video0>')
    # reader = imageio.get_reader('/home/xc/Videos/DSC_0813.MOV')

    # if not cap.isOpened():
    #     print("Unable to open camera")
    #     exit(-1)

    for path in imgs:
        
    # while True:
        # import pdb
        # pdb.set_trace()
        # res, img = cap.read()
        # img = reader.get_next_data()
        # img = Image.open(path).convert('RGB')
        img = cv2.imread(path)
        img = imutils.resize(img, width=640)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        res = True
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.2, 0.4, use_cuda)
            print('------')
            draw_img = plot_boxes_cv2(img, bboxes, None, class_names)
            # import pdb
            # pdb.set_trace()
            # print np.shape(draw_img)
            cv2.imshow('cfgfile', img)
            cv2.waitKey(0)
        else:
             print("Unable to read image")
             exit(-1) 

############################################
if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        folder = sys.argv[3]
        demo(cfgfile, weightfile, folder)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
