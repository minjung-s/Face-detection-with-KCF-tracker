import cv2 as cv
import numpy as np
import os
#import pandas


#
# rootdir = "/home/gpuserver/2019_Winter/mj_test/GT/jpg"
# subject = '13-MaleNoGlasses .avi'
imglist = []
imagedir = "/home/owne/scar/mj_test_/simul/simul"
imagelist = os.listdir(imagedir)
imagelist.sort() #frame
for imgfile in imagelist:
    filename, extension = os.path.splitext(imgfile)
    imglist.append(filename)
    
imagelist = imglist
imagelist.sort(key = int)
print(imagelist)


# txtfilename = '%s_%s_eye_dlib.npy' % (subject, imagedirname)
#
# eyelabels = np.load(txtfilepath)

frame_num = 0
frame_num_re = 0
def trackerCallback(x):
    global frame_num
    frame_num = x
def trackerCallback_re(x):
    global frame_num_re
    frame_num_re = x

totalnum_frames = len(imagelist)


cv.namedWindow('show')

cv.createTrackbar('Frame Number', 'show', 0, int(totalnum_frames)-1, trackerCallback)
cv.createTrackbar('Frame Number_re', 'show', 0, int(totalnum_frames)-1, trackerCallback_re)

auto_flag = True

start = 0
end = totalnum_frames-1


while(True):
    frame_num = cv.getTrackbarPos('Frame Number', 'show')
    filename = imagelist[frame_num]
    filename = filename + ".jpg"
    filepath = os.path.join(imagedir, filename)
    box = np.loadtxt('/home/scar/scar/Data_new/ParkJ_GTfin.csv', delimiter=',')
    #eye_GT =  np.loadtxt('/home/owne/scar/mj_test_/Datasets/OurDataset_rere/LeeS_data_re_eye_0.3_GT.csv', delimiter=',')
    frame = cv.imread(filepath)
    frame = cv.resize(frame, dsize=(320, 240), interpolation=cv.INTER_AREA)
    # eyelabel = eyelabels[frame_num]
    cv.putText(frame, 'frame : %d' % (frame_num+1), (20, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    #cv.putText(frame,eye_GT[frame_num], (20, 50),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),3 )

    x = int(box[frame_num, 0])
    y = int(box[frame_num, 1])
    w = int(box[frame_num, 2])
    h = int(box[frame_num, 3])
    cv.putText(frame, 'x : %d' % (x), (0, 160),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv.putText(frame, 'y : %d' % (y), (100, 160),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv.putText(frame, 'w : %d' % (w), (0, 180),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv.putText(frame, 'h : %d' % (h), (100, 180),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    frame_num_re = cv.getTrackbarPos('Frame Number_re', 'show')
    cv.putText(frame, 'frame_re : %d' % (frame_num_re + 1), (120, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    x_re = int(box[frame_num_re, 0])
    y_re = int(box[frame_num_re, 1])
    w_re = int(box[frame_num_re, 2])
    h_re = int(box[frame_num_re, 3])
    cv.putText(frame, 'x : %d' % (x_re), (0, 200),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv.putText(frame, 'y : %d' % (y_re), (100, 200),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv.putText(frame, 'w : %d' % (w_re), (0, 220),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv.putText(frame, 'h : %d' % (h_re ), (100, 220),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    img = cv.rectangle(frame, (x_re, y_re), (w_re + x_re, h_re + y_re), (255, 0, 0), 2)

    # print(idx_frame)
    # print(x, y, w, h)
    img = cv.rectangle(frame, (x, y), (w + x, h + y), (0, 0, 255), 3)
    cv.imshow('show', frame)

    c = cv.waitKey(10)
    if c == ord('q'):
        frame_num += 1
    if c == ord('w'):
        frame_num -= 1
    if c == ord('a'):
        frame_num_re += 1
    if c == ord('s'):
        frame_num_re -= 1
    # if auto_flag is True:
    #     frame_num += 1
    cv.setTrackbarPos('Frame Number', 'show', frame_num)
    cv.setTrackbarPos('Frame Number_re', 'show', frame_num_re)
cv.destroyAllWindows()

