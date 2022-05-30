import os
import pickle
import numpy as np
import cv2
from typing import List

__CURRENT_DIR__ = os.path.dirname(os.path.abspath(__file__))
__PARENT_DIR__ = os.path.dirname(__CURRENT_DIR__)


class FeatureExtraction:
    pca = None

    @staticmethod
    def set_model_path(model_path):
        FeatureExtraction.pca = pickle.load(open(os.path.join(__CURRENT_DIR__, "models", model_path), 'rb'))
        print(f"/{model_path}")

    @staticmethod
    def eigen_faces_features(faces: List[np.ndarray]) -> np.ndarray:
        face_size = (48, 48)
        resized_faces = np.zeros((len(faces), np.prod(face_size)))
        for i, face in enumerate(faces):
            # Resize face to a fixed size then flatten it
            resized_faces[i] = cv2.resize(face, face_size).ravel()

        eigen_faces = FeatureExtraction.pca.transform(resized_faces)
        return eigen_faces
