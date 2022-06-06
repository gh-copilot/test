#!/usr/bin/env python3

import numpy as np
import cv2
import subprocess
import os
import dlib
import sys
from datetime import datetime

face_detector = dlib.get_frontal_face_detector()

np.random.seed(42)
SELECTED_SIZE = 45


def load_dataset(path_to_dataset, output_path, numpy_output_path):
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    
    files = list(os.listdir(path_to_dataset))
    random_files = np.random.choice(files, size=SELECTED_SIZE, replace=False)
    print(f'Dataset Size: {len(files)}')

    for file in files:
        path = os.path.join(path_to_dataset, file)
        try:
            img = cv2.imread(path)
            face = face_detector(img, 0)[0]
            img = img[face.top():face.bottom(), face.left():face.right()]
            cv2.imwrite(os.path.join(output_path, file), img)
            
            if file in random_files:
                img = cv2.resize(img, (48, 48)).ravel()
                np.save(os.path.join(numpy_output_path, file), img)

        except Exception as e:
            print(f"ERROR at '{path}'", e)
        
    print('Finished saving dataset.')


start = datetime.now()
print(f"Loading dataset {sys.argv[1]}")
load_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
print(f"Time Taken: {datetime.now() - start}")
