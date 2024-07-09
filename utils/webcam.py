import mediapipe.python.solutions.hands as Hand
import cv2
import copy
import numpy as np
import itertools
 
class WebCam:
    def __init__(self, min_detection_confidence:float=0.5, min_tracking_confidence:float=0.7):
        self.cam = cv2.VideoCapture(0)
        self.min_detect_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self.model = Hand
        self.model = self.model.Hands(
            max_num_hands=1,
            min_detection_confidence=self.min_detect_conf,
            min_tracking_confidence=self.min_track_conf
        )

    def calc_landmark_list(self, image:np.ndarray, landmarks:np.ndarray) -> list[list[int]]:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point
    
    
    def pre_process_landmark(self,landmark_list) -> np.ndarray:
        def normalize_(n):
            return n / max_value
        
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def run(self) -> list[np.ndarray]:
        # Capture frame-by-frame
        ret, frame = self.cam.read()
        
        if not ret:
            exit()
        frame = cv2.flip(frame, 1)  # Mirror display
        debug_image = copy.deepcopy(frame)

        # Apply model
        results = self.model.process(frame)
        landmarks_set = []
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                landmarks_set.append(pre_processed_landmark_list)
        return {"landmark":landmarks_set, "frame":frame}
                    
        
  