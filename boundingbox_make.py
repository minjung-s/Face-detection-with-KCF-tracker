import os
import cv2 as cv
import numpy as np
import csv
import dms_eam_custom
import time


detector_names = ['GT', 'haar', 'HoG', 'Dlib', 'MTCNN', 'RestrictedMTCNN']
detector_name = detector_names[0]
# make paths and flags of src
dirpaths = ['/media/gpuserver/50A268B4A268A068/Datasets/YawDD/user06/YawDD dataset/Dash/Female',
            '/media/gpuserver/50A268B4A268A068/Datasets/YawDD/user06/YawDD dataset/Dash/Male',
            '/media/gpuserver/50A268B4A268A068/Datasets/NTHU DDD dataset/Evaluation Dataset/004',
            '/media/gpuserver/50A268B4A268A068/Datasets/NTHU DDD dataset/Evaluation Dataset/022',
            '/media/gpuserver/50A268B4A268A068/Datasets/NTHU DDD dataset/Evaluation Dataset/026',
            '/media/gpuserver/50A268B4A268A068/Datasets/NTHU DDD dataset/Evaluation Dataset/030',
            '/media/gpuserver/50A268B4A268A068/Datasets/Cockpit_Dataset2']

flags = ['.avi',
         '.avi',
         '.mp4',
         '.mp4',
         '.mp4',
         '.mp4',
         '.jpg']

srcpaths = []
srcflags = []
vi_index = []

for idx, (dirpath, flag) in enumerate(zip(dirpaths, flags)):
    if flag is not '.jpg':
        tmplist = os.listdir(dirpath)
        tmplist.sort()
        for tmpelement in tmplist:
            filename, extension = os.path.splitext(tmpelement)
            if extension == flag:
                srcpaths.append(os.path.join(dirpath, tmpelement))
                srcflags.append(flag)
                vi_index.append(tmpelement)

    elif flag == '.jpg':
        tmplist = os.listdir(dirpath)
        for tmpelement in tmplist:
            tmppath = os.path.join(dirpath, tmpelement)
            tmp = os.listdir(tmppath)
            srcpaths.append(os.path.join(tmppath, tmp[0]))
            srcflags.append(flag)
            vi_index.append(tmpelement)

# imshow and write csv in loop
detector =dms_eam_custom.Detector()
tracker = dms_eam_custom.Tracker()


savepath = ''
Datasetname = str
for idx, (srcpath, srcflag) in enumerate(zip(srcpaths, srcflags)):
    if srcflag is not '.jpg':
        print('read %s' % srcpath)
        if "NTHU" in srcpath : Datasetname = "NTHU DDD dataset"
        else : Datasetname = "YawDD"
        vc = cv.VideoCapture(srcpath)
        tmpfilename, tmpext = os.path.splitext(srcpath)
        savepath = tmpfilename + '_' + detector_name + 'fin' + '.csv'
        f = open(savepath, 'a', encoding='utf-8')
        wr = csv.writer(f)
        index_det = 0
        index_tra = 0
        time_det_total = 0
        time_tra_total = 0
        bbox = [0, 0, 0, 0]
        while(vc.isOpened()):
            _, frame = vc.read()
            if _ is False:
                break
            #-----do something-----
            #bbox = []
            resized_image = cv.resize(frame, dsize=(320, 240), interpolation=cv.INTER_AREA)
            # start_time_det = time.time()
            detected_face = detector.detect_faces(resized_image)
            # time_det = time.time() - start_time_det
            # index_det += 1
            # time_det_total += time_det
            #
            # start_time_tra = time.time()

            if detected_face is not None:
                tracker.reset(resized_image, detected_face.bbox.flatten().tolist())
                bbox = detected_face.bbox
                x1, y1, x2, y2 = bbox
                #bounding_boxes.append(detected_face['box'])
                wr.writerow([x1, y1, x2-x1, y2-y1])
            else:
                detected_face = tracker.update(resized_image)
                if detected_face is not None:
                    bbox = detected_face.bbox
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = bbox
                #bounding_boxes.append(detected_face['box'])
                wr.writerow([x1, y1, x2-x1, y2-y1])
            #----------------------

            # time_tra = time.time() - start_time_det
            # time_tra_total += time_tra
            # index_tra += 1
            #cv.imshow('frame', frame)
            c = cv.waitKey(1)
            if c == ord('q'):
                break
        vc.release()
        #f.closed()
        # fps_det = index_det / time_det_total
        # if time_tra_total == 0:
        #     fps_tra = 0
        # else:
        #     fps_tra = index_tra / time_tra_total
        print('write %s' % savepath)
        # y = open('fps.csv','a',encoding='utf-8')
        # yw = csv.writer(y)
        # yw.writerow([Datasetname, vi_index[idx], fps_det, fps_tra]) #tnwjd


    elif srcflag is '.jpg':
        print('read  %s' % srcpath)
        imglist = os.listdir(srcpath)
        imglist.sort()
        imglist = [os.path.join(srcpath,x) for x in imglist if x.endswith(srcflag)]
        savepath = srcpath + '_' + detector_name + '.csv'
        f = open(savepath, 'a', encoding='utf-8')
        wr = csv.writer(f)
        # index_det = 0
        # index_tra = 0
        # time_det_total = 0
        # time_tra_total = 0
        for imgpath in imglist:
            frame = cv.imread(imgpath)
            # -----do something-----
            resized_image = cv.resize(src=frame, dsize=(320, 240), interpolation=cv.INTER_AREA)
            # start_time_det = time.time()
            detected_face = detector.detect_faces(resized_image)
            # time_det = time.time() - start_time_det
            # index_det += 1
            # time_det_total += time_det
            #
            # start_time_tra = time.time()

            if detected_face is not None:
                tracker.reset(resized_image, detected_face.bbox.flatten().tolist())
                # time_tra = time.time() - start_time_det
                bbox = detected_face.bbox
                x1, y1, x2, y2 = bbox
                #bounding_boxes.append(detected_face['box'])
                wr.writerow([x1, y1, x2-x1, y2-y1])
            else:
                detected_face = tracker.update(resized_image)
                # time_tra = time.time() - start_time_det
                bbox = detected_face.bbox
                x1, y1, x2, y2 = bbox
                # bounding_boxes.append(detected_face['box'])
                wr.writerow([x1, y1, x2-x1, y2-y1])

            # ----------------------
            # time_tra = time.time() - start_time_det
            # time_tra_total += time_tra
            # index_tra += 1
            #cv.imshow('frame', frame)
            c = cv.waitKey(1)
            if c == ord('q'):
                break
        #f.closed()
        # fps_det = index_det / time_det_total
        # if time_tra_total == 0:
        #     fps_tra = 0
        # else:
        #     fps_tra = index_tra / time_tra_total
        print('write %s' % savepath)
        # y = open('fps.csv', 'a', encoding='utf-8')
        # yw = csv.writer(y)
        # yw.writerow(["Cokpit Dataset2", vi_index[idx], fps_det, fps_tra])
