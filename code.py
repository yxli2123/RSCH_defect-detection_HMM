import cv2
import numpy as np
import pickle
from hmmlearn import hmm
from scipy import signal
from PIL import Image


def probForward(observation_sq, A, B, PI):
    # observation_sq shape: (H*W, num_sequence), e.g. (1556*2048, 12)
    step_num = observation_sq.shape[1]
    step_mat = [np.multiply(PI.T, B.T[observation_sq[:, 0]])]
    for i in range(1, step_num):
        step_mat.append(np.multiply(np.dot(step_mat[i - 1], A), B.T[observation_sq[:, i]]))
    prob_ = np.sum(step_mat[step_num - 1], axis=1)
    return prob_


def createMask(prob_list, val_list):
    mask = []
    for p in range(len(prob_list)):
        temp = np.where(prob_list[p] > val_list[p], 255, 0)
        temp = signal.convolve2d(temp, np.ones((5, 5)), mode='same')
        temp = np.floor(temp)
        mask.append(np.where(temp != 0, True, False))
    mask_ = np.logical_or(mask[0], mask[1])
    mask_ = np.logical_or(mask_, mask[2])
    MASK = np.where(mask_, 255, 0)
    MASK = np.array(MASK, dtype='uint8')
    MASK_V = cv2.medianBlur(MASK, 29)
    MASK_V = np.where(MASK_V != 0, False, True)
    MASK = np.logical_and(MASK, MASK_V)
    MASK = np.where(MASK, 255, 0)
    return np.array(MASK, dtype='uint8')


class restoreVideo():
    def __init__(self, videoPath, modelPath, K=13):
        with open(modelPath, "rb") as file_model:
            hmm_model = pickle.load(file_model)
            self.transMat = hmm_model.transmat_
            self.startProb = hmm_model.startprob_
            self.emissionProb = hmm_model.emissionprob_
        self.videoPath = videoPath
        self.K = K
        cap = cv2.VideoCapture(self.videoPath)
        # make sure these ranges are valid
        videoFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.heightRange = [0, videoHeight]
        self.HEIGHT = videoHeight

        self.widthRange = [0, videoWidth]
        self.WIDTH = videoWidth

        self.frameRange = [0, videoFrame]
        self.frameNum = videoFrame

        cnt = 0
        videoSample = []
        colorVideoSample = []
        while cnt < self.frameNum:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
            success, frame_color = cap.read()
            frame = np.array(cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY))
            frame = frame[0:self.HEIGHT, 0:self.WIDTH]
            videoSample.append(frame)
            colorVideoSample.append(frame_color)
            cnt += 1
        self.videoSample = np.array(videoSample, dtype='uint8')
        self.colorVideoSample = np.array(colorVideoSample, dtype='uint8')

    def createFeature(self, featureType='t'):
        if featureType == 'v':
            vertical = np.diff(self.videoSample[0:self.videoSample.shape[0] - 1, :, :], axis=1)
            return np.insert(vertical, 0, vertical[:, 0, :], axis=1)
        if featureType == 'h':
            horizontal = np.diff(self.videoSample[0:self.videoSample.shape[0] - 1, :, :], axis=2)
            return np.insert(horizontal, 0, horizontal[:, :, 0], axis=2)
        else:
            return np.diff(self.videoSample, axis=0)

    def createProb(self, frame_num, featureType='t'):
        if frame_num < int(self.K / 2):
            feature = self.createFeature(featureType=featureType)[0:self.K, :, :]
        elif frame_num >= self.frameNum - int(self.K / 2) - 1:
            feature = self.createFeature(featureType=featureType)[self.frameNum - self.K - 1:self.frameNum - 1, :, :]
        else:
            feature = self.createFeature(featureType=featureType)[
                      frame_num - int(self.K / 2):frame_num + int(self.K / 2) + 1, :, :]
        featureFlatten = np.reshape(feature, (self.K, self.HEIGHT * self.WIDTH), 'F').T
        prob = []
        for i in range(self.K):
            oneOut = np.delete(featureFlatten, i, axis=1)
            oneOutProb = probForward(oneOut, self.transMat, self.emissionProb, self.startProb)
            prob.append(oneOutProb)
        prob = np.array(prob).T

        if frame_num < self.K / 2:
            prob = prob[:, frame_num] / prob.mean(axis=1)
        elif frame_num >= self.frameNum - int(self.K / 2) - 1:
            prob = prob[:, self.K + frame_num - self.frameNum] / prob.mean(axis=1)
        else:
            prob = prob[:, int(self.K / 2)] / prob.mean(axis=1)
        probability = np.reshape(prob, (self.HEIGHT, self.WIDTH), 'F')

        return probability

    def restore(self, batch_num, t_val, v_val, h_val):
        for frame in range(0, self.frameNum):
            Probability_t = self.createProb(frame, 't')
            Probability_v = self.createProb(frame, 'v')
            Probability_h = self.createProb(frame, 'h')
            num = format(frame, '03d')
            mask_frame = createMask([Probability_t, Probability_v, Probability_h], [t_val, v_val, h_val])
            centerFrame = self.colorVideoSample[frame, :, :]
            mask_frame_ = Image.fromarray(mask_frame)
            mask_frame_ = mask_frame_.convert('1')
            mask_frame_.save("./mask/" + batch_num + num + ".jpg")


for i in range(8,9):
    num = format(i, '03d')
    demo = restoreVideo("./" + num + ".mp4", "../../model.pkl")
    demo.restore(num, 4.9, 6.2, 6.2)

