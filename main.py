import os
import cv2 as cv
import numpy as np
import argparse
import detectors_factory

parser = argparse.ArgumentParser()
parser.add_argument('--detector', help='DETECTOR should be one of "haar,haar2,LBP,LBP2,OpenCVDNN,HoG,DlibDNN,MTCNN"')
args = parser.parse_args()

detector_names = ['haar','haar2','LBP','LBP2','OpenCVDNN','HoG','DlibDNN','MTCNN']



detector_name = ''

if args.detector in detector_names:
    detector_name = args.detector
else:
    parser.print_help()
    exit()

detector = detectors_factory.get_detector(detector_name)

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
for idx, (dirpath, flag) in enumerate(zip(dirpaths, flags)):
    if flag is not '.jpg':
        tmplist = os.listdir(dirpath)
        tmplist.sort()
        for tmpelement in tmplist:
            filename, extension = os.path.splitext(tmpelement)
            if extension == flag:
                srcpaths.append(os.path.join(dirpath, tmpelement))
                srcflags.append(flag)
    elif flag == '.jpg':
        tmplist = os.listdir(dirpath)
        for tmpelement in tmplist:
            tmppath = os.path.join(dirpath, tmpelement)
            tmp = os.listdir(tmppath)
            srcpaths.append(os.path.join(tmppath, tmp[0]))
            srcflags.append(flag)

# imshow and write csv in loop

savepath = ''
for idx, (srcpath, srcflag) in enumerate(zip(srcpaths, srcflags)):
    if srcflag is not '.jpg':
        print('read %s' % srcpath)
        vc = cv.VideoCapture(srcpath)
        while(True):
            _, frame = vc.read()
            frame = cv.resize(frame,dsize=(320, 240), interpolation=cv.INTER_AREA)
            #-----do something-----
            result = detector.detect(image=frame)

            print(type(result), result)
            #----------------------

            # cv.imshow('frame', frame)
            # c = cv.waitKey(1)
            # if c == ord('q'):
            #     break
        vc.release()
        tmpfilename, tmpext = os.path.splitext(srcpath)
        savepath = tmpfilename + '_' + detector_name + '.csv'
        print('write %s' % savepath)
    elif srcflag is '.jpg':
        print('read  %s' % srcpath)
        imglist = os.listdir(srcpath)
        imglist.sort()
        imglist = [os.path.join(srcpath,x) for x in imglist if x.endswith(srcflag)]
        for imgpath in imglist:
            frame = cv.imread(imgpath)
            frame = cv.resize(frame, dsize=(320, 240), interpolation=cv.INTER_AREA)
            # -----do something-----
            result = detector.detect(image=frame)
            print(type(result), result)
            # ----------------------

            # cv.imshow('frame', frame)
            # c = cv.waitKey(1)
            # if c == ord('q'):
            #     break
        savepath = srcpath + '_' + detector_name + '.csv'
        print('write %s' % savepath)
