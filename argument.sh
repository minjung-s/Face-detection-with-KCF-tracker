#!/bin/bash

for i in 1 2 3 4
python3 main.py --detector OpenCVDNN
python3 main.py --detector HoG
python3 main.py --detector DlibDNN
python3 main.py --detector MTCNN
done


