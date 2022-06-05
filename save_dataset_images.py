#!/usr/bin/env python3

import numpy as np
import cv2
import subprocess
import os
import dlib
import sys
from datetime import datetime

face_detector = dlib.get_frontal_face_detector()


def get_files_count(directory):
    return int(
        subprocess.check_output(f"find '{directory}' -type f | wc -l", shell=True, stderr=subprocess.STDOUT, text=True))

def load_dataset(path_to_dataset):
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)

    dataset_size = get_files_count(path_to_dataset)
    print(f'Dataset Size: {dataset_size}')
    for root, dirs, files in os.walk(path_to_dataset):
        for file in files:
            if file.lower().endswith(".avi"):
                continue
            path = os.path.join(root, file)
            try:
                img = cv2.imread(path)
                face = face_detector(img, 0)[0]
                img = img[face.top():face.bottom(), face.left():face.right()]
                cv2.imwrite(path, img)
            except Exception as e:
                print(f"ERROR at '{path}'", e)
    print('Finished saving dataset.')


start = datetime.now()
print(f"Loading dataset {sys.argv[1]}")
load_dataset(sys.argv[1])
print(f"Time Taken: {datetime.now() - start}")
