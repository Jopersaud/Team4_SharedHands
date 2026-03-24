import numpy as np
import cv2
import mediapipe as mp
import os
from pathlib import Path


video_Path = Path("data/raw_videos/ASL_dynamic")
# Only goes to the clip frame folders
clip_Dir = list(video_Path.glob("*/*_frames"))

# Test image path
test_Path = Path("data/raw_videos/ASL_dynamic/A/A_clip2_frames")


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
Frame_Files = sorted(list(test_Path.glob("*.jpg")))

cliplm_sequence = []

def load_video_ds():
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(Frame_Files):
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            lmframe_sample = []
            lmlabels = []

            T = 16

            # Print handedness and draw hand landmarks on the image.
            #print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            if results.multi_handedness:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        # Normalizing landmarks relative to the hand geometry by subtracting wrist lm values
                        base_x = hand_landmarks.landmark[0].x
                        base_y = hand_landmarks.landmark[0].y
                        base_z = hand_landmarks.landmark[0].z
                        # add each landmarks data (x, y, z) into sample list
                        lmframe_sample.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                cliplm_sequence.append(lmframe_sample)
        
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))


            # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue

            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                    hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    #Sorts through all the clip frames with visisble hands and chooses 16 frames from each image folder
    #Spreads out the selection of frames taken per clip so that the full gesture + landmarks can be seen as it scans
        N = len(cliplm_sequence)
        if N >= T:
            sample_seq = []
            for k in range(T):
                i = int(k * (N / T))
                sample_seq.append(cliplm_sequence[i])

    #creates numpy array with lists of all the landmark positions for the chosen 16 frames in A_clip2_frames
    #Will store all the gathered lm info into final lm_npdata.npy and lmlabels.npy files. Can feed those directly to model
        lm_npdata = np.array(sample_seq)
        #lmlabels = np.array(lmlabels)
        np.save("lm_npdata.npy", lm_npdata)
        #np.save("lmlabels.npy", lmlabels)


#load_video_ds()
X = np.load("lm_npdata.npy")
print(X.shape)

