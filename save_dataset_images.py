#!/usr/bin/env python3

import numpy as np
import cv2
import subprocess
import os
import dlib
import sys
from datetime import datetime
import argparse
import traceback


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--numpy_output', type=str, required=True)
    parser.add_argument('--double', default=False, action="store_true")
    parser.add_argument('--size', type=int, default=45)
    return parser.parse_args()


args = get_args()

face_detector = dlib.get_frontal_face_detector()

np.random.seed(42)
SELECTED_SIZE = args.size + 10


def load_dataset(path_to_dataset, output_path, numpy_output_path):
    path_to_dataset = os.path.join(os.getcwd(), path_to_dataset)
    
    files = list(os.listdir(path_to_dataset))
    random_files = np.random.choice(files, size=SELECTED_SIZE, replace=False)
    print(f'Dataset Size: {len(files)}')

    for file in files:
        path = os.path.join(path_to_dataset, file)
        try:
            img = cv2.imread(path)
            face = face_detector(img, 4)[0]
            img = img[face.top():face.bottom(), face.left():face.right()]
            if args.double:
                face = face_detector(img, 4)[0]
                img = img[face.top():face.bottom(), face.left():face.right()]
            cv2.imwrite(os.path.join(output_path, file), img)
            
            if file in random_files:
                img = cv2.resize(img, (48, 48)).ravel()
                np.save(os.path.join(numpy_output_path, file), img)

        except Exception as e:
            print(f"ERROR at '{path}'", e)
            traceback.print_exc()
        
    print('Finished saving dataset.')


def filter_numpy_dataset(numpy_output_path):
    files = list(os.listdir(numpy_output_path))
    random_files = np.random.choice(files, size=args.size, replace=False)
    for file in files:
        if file not in random_files:
            os.remove(os.path.join(numpy_output_path, file))


start = datetime.now()
print(f"Loading dataset {args.dataset}")
load_dataset(args.dataset, args.output, args.numpy_output)
filter_numpy_dataset(args.numpy_output)
print(f"Time Taken: {datetime.now() - start}")
