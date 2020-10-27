import cv2 as cv
import numpy as np
import os




srcpaths_csv = ['/home/scar/scar/Data_new/LeeS_GTfin.csv',
            '/home/scar/scar/Data_new/LimD_GTfin.csv',
            '/home/scar/scar/Data_new/ParkJ_GTfin.csv',
            '/home/scar/scar/Data_new/ShinM_GTfin.csv']

# make paths and flags of src
dirpaths = [ '/home/scar/scar/Data_new']


flags = ['.mp4']


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
                print(filename)

    elif flag == '.jpg':
        tmplist = os.listdir(dirpath)
        for tmpelement in tmplist:
            tmppath = os.path.join(dirpath, tmpelement)
            tmp = os.listdir(tmppath)
            srcpaths.append(os.path.join(tmppath, tmp[0]))
            srcflags.append(flag)
            vi_index.append(tmpelement)


for idx, (srcpath_csv,srcpath, srcflag) in enumerate(zip(srcpaths_csv,srcpaths, srcflags)):
    if srcflag is not '.jpg':
        vc = cv.VideoCapture(srcpath)
        idx_frame = 0
        box = np.loadtxt(srcpath_csv, delimiter=',')
        print("^^")
        print(srcpath)
        print(srcpaths_csv[idx])
        while(True):
            _, frame = vc.read()
            if frame is None:
                break

            #print('read %s' % srcpath)
            frame = cv.resize(frame,dsize=(320, 240), interpolation=cv.INTER_AREA)
            #box = np.fromstring(idx_frame, dtype=float, sep=',')
            #x, y, w, h = box[idx_frame][]
            x = int(box[idx_frame,0])
            y = int(box[idx_frame,1])
            w = int(box[idx_frame,2])
            h = int(box[idx_frame,3])

            img = cv.rectangle(frame, (x,y), (w+x,h+y), (0, 0, 255), 3)
            print(x, y)
            cv.imshow('image', frame)
            c = cv.waitKey(1)
            if c == ord('q'):
                break
            #cv.destroyAllWindows()
            idx_frame += 1
        vc.release()
        # tmpfilename, tmpext = os.path.splitext(srcpath)
        # savepath = tmpfilename + '_' + detector_name + '.csv'
        # print('write %s' % savepath)
    elif srcflag is '.jpg':
        print('read  %s' % srcpath)
        idx_frame = 0
        imglist = os.listdir(srcpath)
        imglist.sort()
        imglist = [os.path.join(srcpath,x) for x in imglist if x.endswith(srcflag)]
        box = np.loadtxt(srcpaths_csv[idx], delimiter=',')
        for imgpath in imglist:
            frame = cv.imread(imgpath)
            if frame is None:
                break
            frame = cv.resize(frame, dsize=(320, 240), interpolation=cv.INTER_AREA)
            # x, y, w, h = box[idx_frame][]
            x = int(box[idx_frame, 0])
            y = int(box[idx_frame, 1])
            w = int(box[idx_frame, 2])
            h = int(box[idx_frame, 3])
            img = cv.rectangle(frame, (x, y), (w, h), (0, 0, 255), 3)
            cv.imshow('image', frame)
            # print(idx_frame)
            c = cv.waitKey(1)
            if c == ord('q'):
                break
            idx_frame += 1
