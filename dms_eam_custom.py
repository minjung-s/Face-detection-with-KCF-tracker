import os
import numpy as np
import cv2 as cv
#from  import HoG
#from  import Haar-Cascade
from mtcnn.mtcnn import MTCNN
import configparser
import ast
import argparse
import csv
#from tensorflow.keras.models import model_from_json
#from tensorflow.keras.optimizers import Adam


ap = argparse.ArgumentParser()
ap.add_argument("D", type=str,
	help="detector model: haar or hog or mtcnn or dlib")
args = ap.parse_args()
detector = args.D

def crop_image(image=None, bbox=None, crop_size=(160, 160), crop_ratio=0.1):
    cropped_image = None
    if image is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0:
            return None
        if x2 > (image.shape[1] - 1) or y2 > (image.shape[0] - 1):
            return None
        if x1 >= x2 or y1 >= y2:
            return None
        if image.shape[1] <= 0 or image.shape[0] <= 0:
            return None
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        m_x = int(w * crop_ratio)
        m_y = int(h * crop_ratio)
        crop_bbox = list(map(int, [bbox[0] - m_x, bbox[1] - m_y, bbox[2] + m_x, bbox[3] + m_y]))
        x1, y1, x2, y2 = crop_bbox
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            cropped_image = np.pad(image, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - image.shape[0], 0)),
                                           (np.abs(np.minimum(0, x1)), np.maximum(x2 - image.shape[1], 0)), (0, 0)),
                                   mode="constant")
            y1 += np.abs(np.minimum(0, y1))
            y2 += np.abs(np.minimum(0, y1))
            x1 += np.abs(np.minimum(0, x1))
            x2 += np.abs(np.minimum(0, x1))
        else:
            cropped_image = np.copy(image)
        cropped_image = cropped_image[y1:y2, x1:x2, :]
        cropped_image = cv.resize(cropped_image, (crop_size[0], crop_size[1]), interpolation=cv.INTER_CUBIC)
    return cropped_image



class Face:
    def __init__(self):
        self.bbox = None        # ndarray
        self.image = None       # ndarray
        self.confidence = None  # float
        self.emotion = None     # ndarray


class Detector:
    def __init__(self, config_filename='./DMS_EAM/eam_run_config.ini'):
        if os.path.exists(config_filename):
            config = configparser.ConfigParser()
            config.read(config_filename)
            self.model_name = config.get('Detector', 'ModelName')
            self.model_path = config.get('Detector', 'ModelPath')
            self.face_crop_size = config.getint('Detector', 'FaceCropSize')
            self.face_crop_size = [self.face_crop_size, self.face_crop_size]
            self.face_crop_ratio = config.getfloat('Detector', 'FaceCropRatio')
            self.face_size_threshold = config.getfloat('Detector', 'FaceSizeThreshold')
            self.face_confidence_threshold = config.getfloat('Detector', 'FaceConfidenceThreshold')
        else:
            self.model_name = detector
            self.model_path = ''
            self.face_crop_size = [160, 160]
            self.face_crop_ratio = 0.1
            self.face_size_threshold = 0.0
            self.face_confidence_threshold = 0.9

        if detector == "mtcnn" or "MTCNN":
            self.detector = MTCNN()
        elif detector == "haar" or "Haar" or "Haar-cascade":
            self.detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
			#
        elif detector == "dlib" or "Dlib" :
            self.detector = Dlib()
        elif detector == "hog" or "HoG":
            self.detector = HoG()

    # Parameters:   image: ndarray
    # Returns:      detected_face: object(Face) or None
    def detect_faces(self, image=None):
        detected_face = None
        #x1, y1, w, h = None
        if image is not None and self.detector is not None:
            det = self.detector.detect_faces(image)
            candidate_faces = []
            for d in det:
                face = Face()
                x1, y1, w, h = d['box']
                x2, y2 = x1 + w, y1 + h
                face.bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
                face.confidence = d['confidence']
                candidate_faces.append(face)
            detected_face = self.filter_faces(candidate_faces)
            if detected_face is not None:
                x1, y1, x2, y2 = detected_face.bbox
                if x1 < 0 or y1 < 0:
                    return None
                if x2 > (image.shape[1] - 1) or y2 > (image.shape[0] - 1):
                    return None
                if x1 >= x2 or y1 >= y2:
                    return None
                if image.shape[1] <= 0 or image.shape[0] <= 0:
                    return None
                detected_face.image = crop_image(image=image, bbox=detected_face.bbox,
                                                 crop_size=self.face_crop_size, crop_ratio=self.face_crop_ratio)
        return detected_face

    # Parameters:   faces: list(object(Face))
    # Returns:      filtered_face: object(Face) or None
    def filter_faces(self, faces):
        filtered_face = None
        if isinstance(faces, list):
            if len(faces) > 0:
                indexes = []
                indexes = indexes + [i for i, f in enumerate(faces) if f.confidence >= self.face_confidence_threshold]
                if len(indexes) > 0:
                    candidate_faces = []
                    candidate_faces = candidate_faces + [faces[i] for i in indexes]
                    area = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in candidate_faces]
                    max_index = np.argmax(area)
                    filtered_face = candidate_faces[int(max_index)]
        return filtered_face


class Tracker:
    def __init__(self, config_filename='./DMS_EAM/eam_run_config.ini'):
        if os.path.exists(config_filename):
            config = configparser.ConfigParser()
            config.read(config_filename)
            self.model_name = config.get('Tracker', 'ModelName')
            self.model_path = config.get('Tracker', 'ModelPath')
            self.face_crop_size = config.getint('Tracker', 'FaceCropSize')
            self.face_crop_size = [self.face_crop_size, self.face_crop_size]
            self.face_crop_ratio = config.getfloat('Tracker', 'FaceCropRatio')
        else:
            self.model_name = 'kcf'
            self.model_path = ''
            self.face_crop_size = [160, 160]
            self.face_crop_ratio = 0.1
        self.tracker = cv.TrackerKCF_create()

    # Parameters:   image: ndarray
    # Returns:      tracked_face: object(Face) or None
    # def update(self, image=None):
    #     tracked_face = None
    #     if image is not None and self.tracker is not None:
    #         ok, tracked_bbox = self.tracker.update(image)
    #         if ok:
    #             tracked_face = Face()
    #             x1, y1, x2, y2 = tracked_bbox
    #             tracked_face.bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
    #             tracked_face.confidence = 1.0
    #             tracked_face.image = crop_image(image=image, bbox=tracked_face.bbox,
    #                                             crop_size=self.face_crop_size, crop_ratio=self.face_crop_ratio)
    #     return tracked_face
    def update(self, image=None):
        tracked_face = None
        #x1, y1, w, h = None
        if image is not None and self.tracker is not None:
            ok, tracked_bbox = self.tracker.update(image)
            if ok:
                x1, y1, x2, y2 = tracked_bbox
                w = x2 - x1
                h = y2 - y1
                if x1 < 0 or y1 < 0:
                    return tracked_face
                if x2 > (image.shape[1] - 1) or y2 > (image.shape[0] - 1):
                    return tracked_face
                if x1 >= x2 or y1 >= y2:
                    return tracked_face
                if image.shape[1] <= 0 or image.shape[0] <= 0:
                    return tracked_face
                tracked_face = Face()
                tracked_face.bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
                tracked_face.confidence = 1.0
                tracked_face.image = crop_image(image=image, bbox=tracked_face.bbox,
                                                crop_size=self.face_crop_size, crop_ratio=self.face_crop_ratio)
        return tracked_face

    # Parameters:   image: ndarray
    #               bbox: ndarray
    def reset(self, image=None, bbox=None):
        if image is not None and bbox is not None:
            if self.tracker is not None:
                del self.tracker
            self.tracker = cv.TrackerKCF_create()
            ok = self.tracker.init(image, tuple(bbox))
            if not ok:
                del self.tracker
                self.tracker = None



class EMOTION:
    def __init__(self, config_filename='./DMS_EAM/eam_run_config.ini'):
        if os.path.exists(config_filename):
            config = configparser.ConfigParser()
            config.read(config_filename)
            self.input_width = config.getint('Image', 'InputWidth')
            self.input_height = config.getint('Image', 'InputHeight')
            self.input_channels = config.getint('Image', 'InputChannels')
            self.face_crop_size = config.getint('Image', 'FaceCropSize')
            self.face_crop_ratio = config.getfloat('Image', 'FaceCropRatio')
            self.flip = config.getboolean('Image', 'Flip')
            self.normalize = config.getboolean('Image', 'Normalize')
            self.detector = Detector(config_filename=config_filename)
            self.tracker = Tracker(config_filename=config_filename)
            self.classifier = Classifier(config_filename=config_filename)
        else:
            self.input_width = 320
            self.input_height = 240
            self.input_channels = 3
            self.face_crop_size = 160
            self.face_crop_ratio = 0.1
            self.flip = True
            self.normalize = True
            self.detector = None
            self.tracker = None
            self.classifier = None

    # Parameters:   image: ndarray
    #               mode: str
    # Returns:      bbox: ndarray
    #               prediction: ndarray
    #               max_prediction: int
    #               resized_image: ndarray
    def predict(self, image=None, mode='max'):
        bbox, prediction, max_prediction, resized_image = None, None, None, None
        if image is not None and self.detector is not None and self.tracker is not None and self.classifier is not None:
            resize_scale_factor = self.input_width / image.shape[1]
            resized_image = cv.resize(src=image, dsize=(0, 0), fx=resize_scale_factor, fy=resize_scale_factor,
                                      interpolation=cv.INTER_CUBIC)
            detected_face = self.detector.detect_faces(resized_image)
            if detected_face is not None:
                self.tracker.reset(resized_image, detected_face.bbox.flatten().tolist())
            else:
                detected_face = self.tracker.update(resized_image)
            if detected_face is not None:
                if self.normalize:
                    detected_face.image = detected_face.image / 255.0
                detected_face.emotion = self.classifier.predict(detected_face.image)
                bbox, prediction = detected_face.bbox, detected_face.emotion
                max_prediction = np.argmax(prediction)
        return bbox, prediction, max_prediction, resized_image

    # Parameters:   image: ndarray
    #               mode: str
    # Returns:      prediction: ndarray
    #               max_prediction: int
    #               resized_image: ndarray
    def evaluate(self, image=None, mode='max'):
        prediction, max_prediction, resized_image = None, None, None
        if image is not None and self.classifier is not None:
            resized_image = cv.resize(src=image, dsize=(160, 160), interpolation=cv.INTER_CUBIC)
            normalized_image = resized_image / 255.0 if self.normalize else resized_image
            prediction = self.classifier.predict(normalized_image)
            max_prediction = np.argmax(prediction)
        return prediction, max_prediction, resized_image
