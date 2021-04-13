from __future__ import print_function

import sys
import os
# sys.path.insert(0, 'E:/python35/Lib/site-packages/caffe/python')
import caffe
from caffe.proto import caffe_pb2
# import onnx

caffe.set_mode_cpu()
# from onnx2caffe._transformers import ConvAddFuser, ConstantsToInitializers
# from onnx2caffe._graph import Graph

# import onnx2caffe._operators as cvt
# import onnx2caffe._weightloader as wlr
# from onnx2caffe._error_utils import ErrorHandling
# from onnx import shape_inference
import numpy as np
import cv2

cap = cv2.VideoCapture(r'./resources/Safety_Full_Hat_and_Vest.mp4')

if __name__ == "__main__":
    prototxt_path = "./resources/worker-safety-mobilenet/worker_safety_mobilenet.prototxt"
    caffemodel_path = "./resources/worker-safety-mobilenet/worker_safety_mobilenet.caffemodel"

    # def caffe_test():
    weight_file = caffemodel_path
    model_file = prototxt_path
    net = caffe.Net(model_file, weight_file, caffe.TEST)

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))

        x = 120
        y = 200
        frame = frame[x:x+224, y:y+224, :]
        cv2.imshow('', frame)
        cv2.waitKey(1)
        inframe = frame.transpose((2, 0, 1))

        net.blobs["data"].data[...] = inframe

        y_test = net.forward()["detection_out"][0][0]
        if len(y_test):
            print(y_test)

            for obj_sm in y_test:
                if (obj_sm[1] < 0): continue
                if (obj_sm[2] > 0.4):
                    # Detect safety vest
                    if (int(obj_sm[1])) == 2:
                        xmin_sm = int(obj_sm[3] * 224)
                        ymin_sm = int(obj_sm[4] * 224)
                        xmax_sm = int(obj_sm[5] * 224)
                        ymax_sm = int(obj_sm[6] * 224)
                        cv2.rectangle(frame, (xmin_sm , ymin_sm), (xmax_sm , ymax_sm), (0, 255, 0), 2)
                        cv2.imshow(' ', frame)
                        cv2.waitKey(1)







