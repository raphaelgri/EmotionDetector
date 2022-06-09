import pprint

import numpy
from mtcnn import MTCNN
import cv2

import image_transformer


class FaceExtractor:

    def __init__(self, video_name):
        self._video_name = video_name
        self._wait = 1 # to prevent crashing by cv2

    def extract_faces(self, current_frame = 0, frame_step = 150, end_frame = 1000):
        detector = MTCNN()
        faces = []
        cap = cv2.VideoCapture(self._video_name)
        while cap.isOpened():
            self.frame_jumper(cap, current_frame)
            success, frame = cap.read()
            if success:
                frame = self.to_PIL_colour(frame)
                face = detector.detect_faces(frame)
                if face:
                    cropped_face = self.crop_face(frame, face)
                    cropped_face = self.change_size(cropped_face)
                    faces.append(cropped_face)
                    # cv2.imshow("a", cropped_face)
                    pprint.pprint(face)
                current_frame += frame_step
                cv2.waitKey(self._wait)
                if current_frame > end_frame:
                    cap.release()
                    break
            else:
                cap.release()
                break
        return faces

    def frame_jumper(self, cap, jump_to):
        cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to)

    def to_PIL_colour(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def to_CV2_colour(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def face_coordinates(self, face):
        x = face[0]["box"][0]
        y = face[0]["box"][1]
        width = face[0]["box"][2]
        height = face[0]["box"][3]
        return x, y, width, height

    def crop_face(self, frame, face):
        x, y, width, height = self.face_coordinates(face)
        y1 = y+height
        x1 = x+width
        return frame[y:y1, x:x1]

    def change_size(self, cropped_face, width="later", height="later"):
        transform = image_transformer.TransformImage()
        cropped_face = transform.resize_and_pad(cropped_face)
        cropped_face = self.to_CV2_colour(cropped_face)
        return cropped_face

    def show_face(self, face):
        for img in face:
            cv2.imshow("display", img)
            cv2.waitKey(2000)
        cv2.destroyAllWindows()

