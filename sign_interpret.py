import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model', 'keypoint_classifier'))

import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
import argostranslate.package
import argostranslate.translate
import language_tool_python
from keypoint_classifier import KeyPointClassifier

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Download translation model once
from_code = "en"
to_code = "hi"
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
)
argostranslate.package.install_from_path(package_to_install.download())

class SignInterpreter:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5
        )
        self.classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.labels = [row[0] for row in csv.reader(f)]

        self.tool = language_tool_python.LanguageTool('en-US')
        self.expression = " "
        self.prevchar = ' '
        self.count = 0
        self.spcount = 0
        self.history = []

    def process_frame(self, frame):
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            pre_processed_landmark_list = []

            if len(results.multi_hand_landmarks) == 2:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list += self._pre_process_landmark(landmark_list)
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            elif len(results.multi_hand_landmarks) == 1:
                pre_processed_landmark_list = [0.0]*42
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list += self._pre_process_landmark(landmark_list)
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                pre_processed_landmark_list = [0.0]*84

            hand_sign_id, confidence = self.classifier(pre_processed_landmark_list)
            sign = self.labels[hand_sign_id]

            if confidence >= 0.85:
                if self.prevchar == sign:
                    self.count += 1
                else:
                    self.prevchar = sign
                    self.count = 0

                if self.count >= 10 and self.expression[-1] != sign:
                    self.prevchar = ' '
                    self.count = 0
                    if sign == "fullstop":
                        corrected = self._correct_text(self.expression)
                        translated = argostranslate.translate.translate(corrected, from_code, to_code)
                        self.history.append((corrected, translated))
                        self.expression = " "
                    else:
                        self.expression += sign

            debug_image = self._draw_info_text(debug_image, confidence, sign)

        else:
            self.spcount += 1
            if self.spcount >= 10 and self.expression[-1] != " ":
                self.expression += " "
                self.spcount = 0

        return debug_image, self.expression.strip(), self.history

    def _correct_text(self, text):
        matches = self.tool.check(text)
        return language_tool_python.utils.correct(text, matches)

    def _calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        return [
            [min(int(lm.x * image_width), image_width - 1),
             min(int(lm.y * image_height), image_height - 1)]
            for lm in landmarks.landmark
        ]

    def _pre_process_landmark(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for i in range(len(temp)):
            temp[i][0] -= base_x
            temp[i][1] -= base_y
        flat = list(itertools.chain.from_iterable(temp))
        max_value = max(map(abs, flat)) or 1
        return [val / max_value for val in flat]

    def _draw_info_text(self, image, confidence, hand_sign_text):
        if hand_sign_text and confidence >= 0.7:
            cv.putText(image, hand_sign_text, (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return image
