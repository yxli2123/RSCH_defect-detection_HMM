

# RSCH_defect-detection_HMM

## Model

Hidden Markov Model

- Hidden states:
  - "Background 1"
  - "Transition from background to foreground"
  - "Foreground"
  - "Transition from foreground to background"
  - "Background 2"
- Observasion space, discrete value from 0 to 255:
  - The absolute difference of 2 neighbor frames (t derivative)
  - The absolute difference of 2 neighbor rows (y derivative)
  - The absolute difference of 2 neighbor columns (x derivative)

<img src="image_sample/illustration.png" alt="illustration" style="zoom:50%;" />

## Train

Step 1, extract 3 kinds of features

- Compute the absolute difference of 2 neighbor frames
- Compute the absolute difference of 2 neighbor rows 
- Compute the absolute difference of 2 neighbor columns

Step 2, group frames by 13

- 12 frames ahead of the target frame and 12 frames after the target frame

Step 3, train 3 HMMs using Forward-Backward algorithm



## Detect

Step 1, extract 3 kinds of features

Step 2, group frames by 13

Step 3, calculate likelihood of rest image sequences

- Remove each frame in the image sequences, e.g., remove the 2nd frame of the total 13 frames
- Calculate the likelihood of the rest 12 frames. Ensemble 3 kinds of features
- Normalize the 12 likelihoods. If the likelihood of the center frame is greater than a given threshold, then the center pixel is marked as defects

Step 4, using median filter to mitigate the false alarm

- Moving objects and defects have something in common in the temporal domain, for example they all last in the same location for a very short period of time, but they have difference in spaticial domain. 
- The most noticeable difference between moving objects and defects are that moving objects are larger than the defects. Moreover, moving objects are somehow continues in the temporal domain while defects are random
- Therefore, we use median filter to filter out the small prospective defects and the remaining defects are actually the moving objects.

## Samples

### Sample 1

Before and after

<img src="image_sample/image1_results.jpeg" alt="image1_results" style="zoom:50%;" />

Detection results

<img src="image_sample/image1_detection.jpeg" alt="image1_detection" style="zoom:50%;" />

### Sample 2

Before and after

<img src="image_sample/image2_results.jpeg" alt="image2_results" style="zoom:50%;" /><img src="image_sample/image2_detection.jpeg" alt="image2_detection" style="zoom:0%;" />