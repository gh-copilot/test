#!/usr/bin/env python3

import numpy as np
import cv2
import subprocess
import os
import dlib
import sys
from datetime import datetime

face_detector = dlib.get_frontal_face_detector()

size = int(sys.argv[3]) if len(sys.argv) > 3 else 128

def get_files_count(directory):
    return int(
        subprocess.check_output(f"find '{directory}' -type f | wc -l", shell=True, stderr=subprocess.STDOUT, text=True))

def load_dataset(path_to_dataset):
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)

    dataset_size = get_files_count(path_to_dataset)
    print(f'Dataset Size: {dataset_size}')
    features = np.zeros((dataset_size, size * size), dtype=np.uint8)
    i = 0
    for root, dirs, files in os.walk(path_to_dataset):
        for file in files:
            if file.lower().endswith(".avi"):
                continue
            path = os.path.join(root, file)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                face = face_detector(img, 0)[0]
                img = img[face.top():face.bottom(), face.left():face.right()]
                img = cv2.resize(img, (size, size)).ravel()
                features[i] = img
                i += 1
            except:
                print(f"ERROR at '{path}'")
    print('Finished loading dataset.')
    return features[:i]


start = datetime.now()
print(f"Loading dataset {sys.argv[1]} , image size: {size}x{size}")
data = load_dataset(sys.argv[1])
print("Saving...")
np.save(f"{sys.argv[2]}_{size}x{size}.npy", data)
print("Saved")
print(f"Time Taken: {datetime.now() - start}")
