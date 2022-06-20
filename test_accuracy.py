# Boilerplate to Enable Relative imports when calling the file directly
if (__name__ == '__main__' and __package__ is None) or __package__ == '':
    import sys
    from pathlib import Path

    file = Path(__file__).resolve()
    sys.path.append(str(file.parents[3]))
    __package__ = '.'.join(file.parent.parts[len(file.parents[3].parts):])

import sys
try:
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)
except:
    pass

import os
from feature_extracton import FeatureExtraction
import knn
import cv2
import dlib
import numpy as np
import itertools
from datetime import datetime
import argparse
import math
import traceback


face_detector = dlib.get_frontal_face_detector()
__VERBOSE__ = False
test_dir = ""
load_mode = "numpy"
faces_db = {}
solve_conflicts = None

arr = []
people = {}

def test(number_of_faces, k):
    count = 0
    count_errors = 0

    min_accuracy = 100

    test_dir_contents = sorted(os.listdir(test_dir))
    dir_combinations = itertools.combinations(test_dir_contents, r=number_of_faces)
    print(f"    TestSet Size: {math.comb(len(test_dir_contents), number_of_faces)} - Number of Faces: {number_of_faces} - K: {k}")

    for dirs in dir_combinations:
        local_count = 0
        local_count_errors = 0

        all_files = []
        for d in dirs:
            files = sorted(os.listdir(os.path.join(test_dir, d)))
            all_files += [[os.path.join(d, f) for f in files]]
        
        dirs = [ d.replace("_numpy", "") for d in dirs]
        
        print(*dirs, sep=" vs ", end='') if __VERBOSE__ else None

        face_track = knn.KNNIdentification(k=k, conflict_solving_strategy=solve_conflicts, threshold=99999999)

        for files in zip(*all_files):

            faces = [get_face(f) for f in files]

            ids = face_track.get_ids(faces)
            local_count += 1
            if not np.all(ids == range(number_of_faces)):
                local_count_errors += 1
                if np.max(ids) >= number_of_faces:
                    print(f"================================================================ Created New Class: {ids}")

        count += local_count
        count_errors += local_count_errors

        local_accuracy = 100 - 100 * (local_count_errors / local_count)
        min_accuracy = min(min_accuracy, local_accuracy)
        
        for d in dirs:
            if d not in people:
                people[d] = 0.0
            people[d] += local_accuracy
        
        arr.append((local_accuracy, [*dirs]))

        print(f": {local_accuracy}%", end='') if __VERBOSE__ else None
        print(" [MIN]") if __VERBOSE__ and local_accuracy == min_accuracy else print("")

    print(f"    Accuracy: {100 - 100 * (count_errors / count)}%")
    print(f"    Min Accuracy: {min_accuracy}%")


def get_face(file):
    face = faces_db.get(file, None)
    if face is None:
        face = load_face_numpy(file) if load_mode == "numpy" else load_face(file)
        faces_db[file] = face

    return face


def load_face(file):
    frame_grey = cv2.imread(os.path.join(test_dir, file), cv2.IMREAD_GRAYSCALE)
    face = face_detector(frame_grey)[0]
    face = frame_grey[face.top():face.bottom(), face.left():face.right()]
    return face


def load_face_numpy(file):
    return np.load(os.path.join(test_dir, file))


def save_faces_db_numpy(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file, face in faces_db.items():
        file = os.path.join(output_dir, f"{file}.npy")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.save(file, face)


def get_command_line_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('test_dir', help="Test Set Directory")
    args_parser.add_argument("-n", "--number-of-faces", help="Number of faces", default=2, type=int, metavar='N')
    args_parser.add_argument("-v", "--verbose", help="Show more verbose output", action="store_true")
    args_parser.add_argument("-k", help="Space-separated K values to pass to KNN", nargs="+", default=[5], type=int)
    args_parser.add_argument("-m", "--model", help="Path to the model to use", default="pca_n=50_affectnet.sav")
    args_parser.add_argument("-c", "--conflict-resolution", help="Resolve conflicts", action="store_const", const="min_distance", default=None)

    group = args_parser.add_mutually_exclusive_group()
    group.add_argument("--load-numpy", help="Load numpy files", action="store_true")
    group.add_argument("--save-numpy", help="Save faces numpy files to directory", default=None, metavar="directory")
    return args_parser.parse_args()


def main(k):
    start = datetime.now()
    test(args.number_of_faces, k)
    end = datetime.now()
    print(f"    Total Time: {end - start}")
    print("   ", "=" * 50)


if __name__ == '__main__':
    args = get_command_line_args()
    test_dir = args.test_dir
    load_mode = "numpy" if args.load_numpy else "image"
    __VERBOSE__ = args.verbose
    solve_conflicts = args.conflict_resolution

    FeatureExtraction.set_model_path(args.model)
    for k in args.k:
        try:
            main(k)
        except:
            traceback.print_exc()

    if args.save_numpy:
        print(f"Saving Numpy Dataset to {args.save_numpy}")
        save_faces_db_numpy(args.save_numpy)

    arr = sorted(arr)
    print("==========================================================")
    for accuracy, name in arr:
        print(f"{'%.5f'%accuracy} : {name}")
    
    print("==========================================================")
    people = sorted(people.items(), key=lambda item: item[1])
    for name, accuracy in people:
        print(f"{'%.5f'%(accuracy/(args.number_of_faces-1))}% : {name}")
    
    
    
