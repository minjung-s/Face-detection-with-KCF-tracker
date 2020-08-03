import cv2 as cv
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN

class HaarCascadeDetector:
    def __init__(self):
        self.detector = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale3(image)
        rects = faces[0]
        weights = faces[2]

        idx = np.argmax(weights)

        output = np.reshape(rects[idx], (1,4))
        return output

class HaarCascadeDetector2:
    def __init__(self):
        self.detector = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        self.detector2 = cv.CascadeClassifier('./haarcascades/haarcascade_profileface.xml')

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale3(image)
        rects = faces[0]
        weights = faces[2]

        idx = np.argmax(weights)

        faces2 = self.detector.detectMultiScale3(image)
        rects2 = faces2[0]
        weights2 = faces2[2]

        idx2 = np.argmax(weights2)

        output1 = np.reshape(rects[idx], (1,4))
        output2 = np.reshape(rects2[idx2], (1,4))

        output = np.hstack((output1, output2))

        return output

class LBPCascadeDetector:
    def __init__(self):
        self.detector = cv.CascadeClassifier('./lbpcascades/lbpcascade_frontalface.xml')

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale3(image)
        rects = faces[0]
        weights = faces[2]

        idx = np.argmax(weights)

        output = np.reshape(rects[idx], (1,4))

        return output

class LBPCascadeDetector2:
    def __init__(self):
        self.detector = cv.CascadeClassifier('./lbpcascades/lbpcascade_frontalface.xml')
        self.detector2 = cv.CascadeClassifier('./lbpcascades/lbpcascade_profileface.xml')

    def detect(self, image):
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale3(image)
        rects = faces[0]
        weights = faces[2]

        idx = np.argmax(weights)

        faces2 = self.detector.detectMultiScale3(image)
        rects2 = faces2[0]
        weights2 = faces2[2]

        idx2 = np.argmax(weights2)

        output1 = np.reshape(rects[idx], (1,4))
        output2 = np.reshape(rects2[idx2], (1,4))

        output = np.hstack((output1, output2))

        return output

class OpenCVDNNDetector:
    def __init__(self):
        modelFile = "./face_detector/opencv_face_detector_uint8.pb"
        configFile = "./face_detector/opencv_face_detector.pbtxt"
        self.detector = cv.dnn.readNetFromTensorflow(modelFile, configFile)
        self.threshold = 0.8
        self.witdh = 320
        self.height = 240

    def detect(self, image):
        blob = cv.dnn.blobFromImage(image, 1.0, (self.witdh,self.height), [104,117,123], False, False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        bboxes = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                x1 = int(detections[0, 0, i, 3] * self.witdh)
                y1 = int(detections[0, 0, i, 4] * self.height)
                x2 = int(detections[0, 0, i, 5] * self.witdh)
                y2 = int(detections[0, 0, i, 6] * self.height)
                bboxes.append([x1, y1, x2-x1, y2-y1])
                confidences.append(confidence)

        idx = np.argmax(confidences)

        output = np.reshape(bboxes[idx], (1,4))
        return output

class HoGSVMDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        # faceRects = self.detector(image, 0)
        dets, scores, _ = self.detector.run(image, 1, -1)

        idx = np.argmax(scores)

        faceRect = dets[idx]

        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()

        bbox = [x1, y1, x2-x1, y2-y1]

        output = np.reshape(bbox, (1,4))
        return output

class DlibDNNDetector:
    def __init__(self):
        self.detector = dlib.cnn_face_detection_model_v1("./face_detector/mmod_human_face_detector.dat")

    def detect(self, image):
        faceRects = self.detector(image, 1)

        idx = np.argmax(faceRects.confidence) #check

        faceRect = faceRects[idx]

        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()

        bbox = [x1, y1, x2 - x1, y2 - y1]

        output = np.reshape(bbox, (1, 4))
        return output

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, image):
        results = self.detector.detect_faces(image)

        bounding_boxes = []
        confidences = []
        for result in results:
            bounding_boxes.append(result['box'])
            confidences.append(result['confidence'])

        idx = np.argmax(confidences)

        output = np.reshape(bounding_boxes[idx], (1,4))
        return output



