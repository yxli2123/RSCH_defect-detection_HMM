import cv2
import numpy as np
from hmmlearn import hmm
import pickle


def creatDemo(src, height_range, width_range, sample_range):
    # read the video and get some basic info
    cap = cv2.VideoCapture(src)
    VIDEO_FRAME = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(VIDEO_FRAME)
    if WIDTH < width_range[1] or HEIGHT < height_range[1]:
        print("width or height out of range!")
        return None

    # truncate the video into a demo
    demo = []
    cnt = sample_range[0]
    while cnt < min(sample_range[1], VIDEO_FRAME):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
        success, frame = cap.read()
        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        frame = frame[height_range[0]:height_range[1], width_range[0]:width_range[1]]
        demo.append(frame)
        cnt += 1
    return np.array(demo, dtype='uint8')


sample_num = 1000
height = 500
width = 700
demo = creatDemo('./train1.mp4', [180, 180+height], [320, 320+width], [0, sample_num])
# convert the (sample_num * VIDEO_WIDTH * VIDEO_HEIGHT) array to observation sequence

# use absolute value of temporal differences
feature = abs(np.diff(demo, axis=0))
feature = np.array(feature, dtype='uint8')

# Reshape the video demo into linear sequence
observation_sq = np.reshape(feature, ((sample_num - 1) * height * width, 1), 'F')

# Set the lengths for HMM learning
lengths = (sample_num - 1) * np.ones(height * width, dtype='uint8')
lengths = lengths.tolist()

# initialize some parameter for a HMM model

model = hmm.MultinomialHMM(n_components=5)
print("begin to fit")
# train this model using observation sequence
model.fit(observation_sq, lengths)
print("fitting finished")
# write the HMM to a file for next use
with open("./HMM_model_train2.pkl", "wb") as file:
    pickle.dump(model, file)
