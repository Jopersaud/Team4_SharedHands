import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

video_Path = Path("data/raw_videos/ASL_dynamic")
# Only goes to the clip frame folders
clip_Dir = sorted(video_Path.glob("*/*_frames"))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def load_video_ds():
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        lmlabels = []
        clip_samples = []
        T = 16
        for clip_path in clip_Dir:
            label = clip_path.parent.name
            Frame_Files = sorted(clip_path.glob("*.jpg"))
            cliplm_sequence = []
            for idx, file in enumerate(Frame_Files):
                # Read an image, flip it around y-axis for correct handedness output
                image = cv2.flip(cv2.imread(file), 1)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                lmframe_sample = []

                # Print handedness and draw hand landmarks on the image.
                #print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()

                if results.multi_hand_landmarks:
                    first_hand = results.multi_hand_landmarks[0]
                    #Normalizing landmarks relative to the hand geometry by subtracting wrist lm values
                    base_x = first_hand.landmark[0].x
                    base_y = first_hand.landmark[0].y
                    base_z = first_hand.landmark[0].z
                    for lm in first_hand.landmark:
                        # add each landmarks data (x, y, z) into sample list
                        lmframe_sample.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    cliplm_sequence.append(lmframe_sample)


    #Sorts through all the clip frames with visisble hands and chooses 16 frames from each image folder
    #Spreads out the selection of frames taken per clip so that the full gesture + landmarks can be seen as it scans
            N = len(cliplm_sequence)
            if N >= T:
                sample_seq = []
                for k in range(T):
                    i = int(k * (N / T))
                    sample_seq.append(cliplm_sequence[i])
                clip_samples.append(sample_seq)
                lmlabels.append(label)

            print(f"Processing clip: {clip_path}")

    #Creates numpy array with lists of all the landmark positions for the chosen 16 frames in A_clip2_frames
    #Will store all the gathered lm info into final lm_npdata.npy and lmlabels.npy files. Can feed those directly to model
    lm_npdata = np.array(clip_samples)
    lm_nplabels = np.array(lmlabels)
    np.save("lm_npdata.npy", lm_npdata)
    np.save("lmlabels.npy", lm_nplabels)

#Runs the conversion function
load_video_ds()

#Variables to access the Numpy files
X = np.load("lm_npdata.npy")
Y = np.load("lmlabels.npy")


