# Face-detection-with-KCF-tracker
('haar','LBP','OpenCVDNN','HoG','DlibDNN','MTCNN') +KCF tracker로 face detection하기

## REQUIREMENT
* Python>=3.4 
* Tensorflow>=1.14 
* OpenCV>=4.1 
* Keras>=2.0.0
* MTCNN
```sh
    $ pip install mtcnn
    # reference : https://github.com/ipazc/mtcnn#mtcnn

```
## Demo


![녹화_2020_10_23_18_10_15_157](https://user-images.githubusercontent.com/41895063/96983286-0da42300-155b-11eb-9c18-07b595ba8419.gif)

    
## USAGE

boundingbox_make.py : face detection하고 싶은 image, video가 있는 dir로 수정

        dirpaths = ['write face dataset dir 1',
                    'write face dataset dir 2',
                    'write face dataset dir 3',
                    '....']

        flags = ['write face dataset 1 file extension ex).avi',
                 'write face dataset 2 file extension ex).mp4',
                 'write face dataset 3 file extension ex).jpg',
                 '....']
                 
Argument : face detection할 알고리즘 선택


```sh
$  boundingbox_make.py D "알고리즘"
ex)
$  boundingbox_make.py D MTCNN
"""
haar :"haar" or "Haar" or "Haar-cascade"
HoG : "hog" or "HoG"
DlibDNN : "dlib" or "Dlib"
MTCNN :"mtcnn" or "MTCNN"
"""
```

## Track bar
