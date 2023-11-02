import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import uuid

class GaitAnalysis:
    def __init__(self, video_path, model_path="./model/pose_landmarker_heavy.task"):
        self.video_path = video_path
        self.model_path = model_path
        self.pose_landmarker_options = self.initialize_landmarker()
        self.frame_rate = None

    def initialize_landmarker(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # These will draw the landmarks on a detect person, as well as the expected connections between those markers.
    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    # Save the annotated frames as a video
    @staticmethod
    def save_annotated_video(frames, frame_rate):
        output_directory = "output_videos"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_video_filename = uuid.uuid4().hex    
    
        output_video_path = os.path.join(output_directory, output_video_filename+".webm")

        if len(frames[0].shape) == 2:  # Check if grayscale
            height, width = frames[0].shape
            # Convert grayscale frame to BGR
            frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames]
        else:
            height, width, _ = frames[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        return output_video_path
    
    # Gap-fill using cubic spline interpolation
    @staticmethod
    def gap_fill(dist_left, dist_right):
        x = np.arange(len(dist_left))
        interp_func_left = interp1d(x, dist_left, kind='cubic', fill_value="extrapolate")
        interp_func_right = interp1d(x, dist_right, kind='cubic', fill_value="extrapolate")
        dist_left_filled = interp_func_left(x)
        dist_right_filled = interp_func_right(x)
        return dist_left_filled, dist_right_filled
               
    # Butterworth low-pass filter
    @staticmethod
    def butterworth_low_pass_filter(dist_left_filled, dist_right_filled, frame_rate):
        fs=len(dist_left_filled)/frame_rate
        nyq = 0.5 * fs
        cutoff = 0.1752  # Using the provided cutoff frequency
        order = 10
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        dist_left_filtered = filtfilt(b, a, dist_left_filled)
        dist_right_filtered = filtfilt(b, a, dist_right_filled)
        return dist_left_filtered, dist_right_filtered

    # Process video and calculate gait
    def process_video(self):
        with self.pose_landmarker_options as landmarker:
            cap = cv2.VideoCapture(self.video_path)          
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))            
            frame_number = 0
            annotated_frames = []
            dist_left, dist_right = [], []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if no more frames are available

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                
                # Calculate the timestamp for the current frame
                frame_timestamp_ms = int(frame_number * (1000 / frame_rate))

                # Perform pose landmarking on the provided single image.
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                annotated_image = self.draw_landmarks_on_image(frame, pose_landmarker_result)
                annotated_frames.append(annotated_image)

                if pose_landmarker_result.pose_landmarks:
                    landmarks = pose_landmarker_result.pose_landmarks[0]
                    keypoint_data = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
                    
                    # Get keypoints and their visibility
                    left_hip = np.array(keypoint_data[23])
                    right_hip = np.array(keypoint_data[24])
                    left_foot_index = np.array(keypoint_data[31])
                    right_foot_index = np.array(keypoint_data[32])

                    dist_left.append(np.linalg.norm(np.subtract(left_hip, left_foot_index)))
                    dist_right.append(np.linalg.norm(np.subtract(right_hip, right_foot_index)))

                frame_number += 1

            cap.release()

            dist_left_filled, dist_right_filled = self.gap_fill(dist_left, dist_right)

            dist_left_filtered, dist_right_filtered = self.butterworth_low_pass_filter(dist_left_filled, dist_right_filled, frame_rate)

            # Find peaks for heel strike
            peaks_left, _ = find_peaks(dist_left_filtered, distance=0.8*frame_rate)
            peaks_right, _ = find_peaks(dist_right_filtered, distance=0.8*frame_rate)

            # Find minima for toe-off
            minima_left, _ = find_peaks(-dist_left_filtered, distance=0.8*frame_rate)
            minima_right, _ = find_peaks(-dist_right_filtered, distance=0.8*frame_rate)

            # Plotting distances, peaks and minima
            # For Left Leg
            plt.figure(1,figsize=(15, 6))
            plt.plot(dist_left_filtered, label="Distances Left Leg", color="blue")
            plt.scatter(peaks_left, [dist_left_filtered[i] for i in peaks_left], color="red", label="Peaks (Heel Strikes) Left Leg")
            plt.scatter(minima_left, [dist_left_filtered[i] for i in minima_left], color="green", label="Minima (Toe-offs) Left Leg")
            plt.title("Distances, Peaks (Heel Strikes), and Minima (Toe-offs) for Left Leg")
            plt.xlabel("Frame Number")
            plt.ylabel("Distance")
            plt.legend()
            plt.grid(True)

            # For Right Leg
            plt.figure(2,figsize=(15, 6))
            plt.plot(dist_right_filtered, label="Distances Right Leg", color="blue")
            plt.scatter(peaks_right, [dist_right_filtered[i] for i in peaks_right], color="red", label="Peaks (Heel Strikes) Right Leg")
            plt.scatter(minima_right, [dist_right_filtered[i] for i in minima_right], color="green", label="Minima (Toe-offs) Right Leg")
            plt.title("Distances, Peaks (Heel Strikes), and Minima (Toe-offs) for Right Leg")
            plt.xlabel("Frame Number")
            plt.ylabel("Distance")
            plt.legend()
            plt.grid(True)

            # Calculate stance times for the right leg
            stance_times_right = []
            for i in range(len(peaks_right)):
                # Find the subsequent toe-off after the current heel strike
                subsequent_minima = [minima for minima in minima_right if minima > peaks_right[i]]
                
                # If there is a subsequent toe-off, calculate stance time
                if subsequent_minima:
                    stance_time = (subsequent_minima[0] - peaks_right[i]) / frame_rate
                    stance_times_right.append(stance_time)

            # Calculate stance times for the left leg
            stance_times_left = []
            for i in range(len(peaks_left)):
                # Find the subsequent toe-off after the current heel strike
                subsequent_minima = [minima for minima in minima_left if minima > peaks_left[i]]
                
                # If there is a subsequent toe-off, calculate stance time
                if subsequent_minima:
                    stance_time = (subsequent_minima[0] - peaks_left[i]) / frame_rate
                    stance_times_left.append(stance_time)
            # Swing Time for left foot
            try:
                swing_time_left = [(peaks_left[i+1] - minima_left[i]) / frame_rate for i in range(len(minima_left) - 1)]
            except IndexError:
                swing_time_left = [(peaks_left[i+1] - minima_left[i]) / frame_rate for i in range(min(len(peaks_left)-1, len(minima_left)))]

            # Swing Time for right foot
            try:
                swing_time_right = [(peaks_right[i+1] - minima_right[i]) / frame_rate for i in range(len(minima_right) - 1)]
            except IndexError:
                swing_time_right = [(peaks_right[i+1] - minima_right[i]) / frame_rate for i in range(min(len(peaks_right)-1, len(minima_right)))]

            # Step Time for left foot
            try:
                step_time_left = [(peaks_left[i+1] - peaks_left[i]) / frame_rate for i in range(len(peaks_left) - 1)]
            except IndexError:
                step_time_left = [(peaks_left[i+1] - peaks_left[i]) / frame_rate for i in range(len(peaks_left) - 2)]

            # Step Time for right foot
            try:
                step_time_right = [(peaks_right[i+1] - peaks_right[i]) / frame_rate for i in range(len(peaks_right) - 1)]
            except IndexError:
                step_time_right = [(peaks_right[i+1] - peaks_right[i]) / frame_rate for i in range(len(peaks_right) - 2)]

            # Double Support Time (heel strike of left foot to toe-off of the right foot)
            double_support_times_left = []  # between left heel strike and right toe-off
            double_support_times_right = []  # between right heel strike and left toe-off

            # Calculate double support time starting from left heel strike to right toe-off
            for i in range(len(peaks_left) - 1):
                # Find the first right toe-off after the current left heel strike
                subsequent_right_toe_off = [m for m in minima_right if m > peaks_left[i]]
                if subsequent_right_toe_off:
                    double_support_duration_left = (subsequent_right_toe_off[0] - peaks_left[i]) / frame_rate
                    double_support_times_left.append(double_support_duration_left)

            # Calculate double support time starting from right heel strike to left toe-off
            for i in range(len(peaks_right) - 1):
                # Find the first left toe-off after the current right heel strike
                subsequent_left_toe_off = [m for m in minima_left if m > peaks_right[i]]
                if subsequent_left_toe_off:
                    double_support_duration_right = (subsequent_left_toe_off[0] - peaks_right[i]) / frame_rate
                    double_support_times_right.append(double_support_duration_right)
            
            # Pad shorter lists with NaN
            max_len = max(len(stance_times_left), len(stance_times_right), 
                        len(swing_time_left), len(swing_time_right),
                        len(step_time_left), len(step_time_right),
                        len(double_support_times_left), len(double_support_times_right))
            
            def pad_list(lst, max_len, pad_value=np.nan):
                return lst + [pad_value] * (max_len - len(lst))

            # Save results to a dataframe
            self.df = pd.DataFrame({
                'Stance Time Left': pad_list(stance_times_left, max_len),
                'Stance Time Right': pad_list(stance_times_right, max_len),
                'Swing Time Left': pad_list(swing_time_left, max_len),
                'Swing Time Right': pad_list(swing_time_right, max_len),
                'Step Time Left': pad_list(step_time_left, max_len),
                'Step Time Right': pad_list(step_time_right, max_len),
                'Double Support Times Left': pad_list(double_support_times_left, max_len),
                'Double Support Times Right': pad_list(double_support_times_right, max_len),
            })

            # Store the results in string
            result = "Stance Time Left: {stance_time_left}, Stance Time Right: {stance_time_right}, Swing Time Left: {swing_time_left}, Swing Time Right: {swing_time_right}, Step Time Left: {step_time_left}, Step Time Right: {step_time_right}, Double Support Times Left: {double_support_times_left}, Double Support Times Right: {double_support_times_right}".format(
                stance_time_left = stance_times_left,
                swing_time_left = swing_time_left,
                stance_time_right = stance_times_right,
                swing_time_right = swing_time_right,
                step_time_left = step_time_left,
                step_time_right = step_time_right,
                double_support_times_left = double_support_times_left,
                double_support_times_right = double_support_times_right,
            )

        output_video_path = self.save_annotated_video(annotated_frames, frame_rate)

        return output_video_path, self.df, result, plt


        
        
