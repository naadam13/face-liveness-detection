# Face Liveness Detection
## Project Overview
We have implemented face verification detection with liveness detection mechanism (to check whether the person detected on the camera is a REAL person or FAKE (eg. image, video, etc. of that person)), built with Convolutional Neural Network.

## Packages and Tools
Check out requirements.txt for the correct version of packages.
- Python 3.9.5
- OpenCV
- TensorFlow 2.5
- Scikit-learn
- Face_recognition
- Flask
- SQLite
- SQLAlchemy (for Flask)

## Files explanation 
* **`collect_dataset.py`**: Collect face in each frame from a *video* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)  
  Command line argument:
  * `--input (or -i)` Path to input input video
  * `--output (or -o)` Path/Directory to output directory of cropped face images
  * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
  * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)
  * `--skip (or -s)` Number of frames to skip before applying face detection and crop (default is 16). The main idea for this is the consequence frames usually give the same face to the dataset, so it can easily causes overfitting and is not a useful data for training.  
  **Example**: example for *fake video* dataset -> `python collect_dataset.py -i videos/fake_1.mp4 -o dataset/fake -d face_detector -c 0.5 -s 15` | example for *real video* dataset -> `python collect_dataset.py -i videos/real_1.mp4 -o dataset/real -d face_detector -c 0.5 -s 15`
* **`face_from_image.py`**: Collect face in each frame from a *image* dataset (real/fake) using face detector model (resnet-10 SSD in this case) and save to a directory (we provided a video example in videos folder, so you can collect the correct video dataset to train the model)  
  Command line argument:
  * `--input (or -i)` Path to input input image (A single image | Since we mainly collect dataset from videos, we use this code only to collect face from those solid-printed picture (picture from paper/card) and we didn't have many of them. So, we make the code just for collect face from 1 image. Feel free to adjust the code if you want to make it able to collect faces from all image in a folder/directory)
  * `--output (or -o)` Path/Directory to output directory of cropped face images
  * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
  * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)  
  **Example**: example for *fake image* dataset -> `python face_from_image.py -i images/fakes/2.jpg -o dataset/fake -d face_detector -c 0.5` | example for *real image* dataset -> `python face_from_image.py -i images/reals/1.jpg -o dataset/real -d face_detector -c 0.5`
* **`livenessnet.py`**: Model architecture for our liveness detection model and build function to build the neural network (there is no command line arguement for this file (no need to do that)). The class *LivenessNet* will be called from the `train_model.py` file in order to build a model and run the training process.
* **`train_model.py`**: The code used to train the liveness detection model and output .model, label_encoder.pickle, and plot.png image files.  
  Command line argument:
  * `--dataset (or -d)` Path to input Dataset
  * `--model (or -m)` Path to output trained model
  * `--le (or -l)` Path to output Label Encoder 
  * `--plot (or -p)` Path to output loss/accuracy plot
  * `--cm (or -c)` Path to confusion matrix  
  **Example**: `python train_model.py -d dataset -m liveness.model -l label_encoder_model.pickle -p plot.png` or `python train_model.py -d dataset -m liveness.h5 -l label_encoder.pickle -p plot.png -c cm.png`
* **`liveness_app.py`**: Run face detection, draw bounding box, and run liveness detection model real-time on webcam  
  Command line argument:
  * `--model (or -m)` Path to trained model
  * `--le (or -l)` Path to Label Encoder
  * `--detector (or -d)` Path to OpenCV\'s deep learning face detector  
  * `--confidence (or -c)` Confidence of face detector model (default is 0.5 | 50%)  
  **Example**: `python liveness_app.py -m liveness.model -l label_encoder_model.pickle -d face_detector -c 0.5` or `python liveness_app.py -m liveness.h5 -l label_encoder.pickle -d face_detector -c 0.5`
**`liveness_app_v2.py`**: Run face detection, draw bounding box, and run liveness detection model real-time on webcam without command line argument
* **`dataset`** folder: Example folder and images for training liveness detection model. (These images are outputs of `collect_dataset.py`)
* **`face_detector`** folder: The folder containing the caffe model files including .prototxt and .caffemodel to use with OpenCV and do face detection
* **`images`** folder: Example folder and images for inputting to `face_from_image.py`
* **`videos`** folder: Example folder and videos for inputting to `collect_dataset.py`

## Basic usage
1. Download/Clone this repo
2. Install the packages in `requirements.txt`
3. Run `liveness_app.py` or `liveness_app_v2.py` or `liveness_app_android.py` or `liveness_app_from_picture.py`
4. That's it!   

## Full Workflow usage and Training your own model
1. Collect video of yourself/others in many light condition (the easiest way to do this is to film yourself/others walking around your/others home) and save to *face_liveness_dection/videos* folder. *The length of the video depends on you.* You don't need to name it with the word 'real' or 'fake'. It's just convention that we found helpful when calling from other codes. **Take a look into that folder, we have dropped some example videos there.**
2. Use those recorded videos and play it on your phone. Then, hold your phone and face the phone screen (running those recorded videos) to the webcam and record your PC/laptop screen. By doing this, you are creating the dataset of someone spoofing the person in the video / pretending to be the person in the video. Try to make sure this new spoofing video has the same length (or nearly) as the original one because we need to avoid *unbalanced dataset*. **Take a look into that folder, we have dropped some example videos there.**
3. Run **`collect_dataset.py`** like the example above in files explanation section for every of your video. Make sure you save the output into the right folder (should be in `dataset` folder and in the right label folder `fake` or `real`). Now you must see a lot of images from your video in the output folder.
4. (Optional, but good to do in order to improve model performance) Take a picture of yourself/others from a paper, photo, cards, etc. and save to *face_liveness_detection/images/fakes*. **Take a look into that folder, we have dropped some example videos there.**
5. If you do step 9, please do this step. Otherwise, you can skip this step. Take more pictures of your face / others' face **in the same amount of the fake images you taken in step 8** and save to *face_liveness_detection/images/reals*. Again, by doing this, we can avoid *unbalanced dataset*. **Take a look into that folder, we have dropped some example videos there.**
6. (Skip this step if you didn't do step 9) Run **`face_from_image.py`** like the example above in files explanation section for every of your image in images folder. Make sure you save the output into the right folder (should be in `dataset` folder and in the right label folder `fake` or `real`). *Note: Like we discussed in files explanation section, you have to run this code 1 image at a time. If you have a lot of images, feel free to adjust the code. So you can run only once for every of your image. (But make sure to save outputs to the right folder)*
7. Run **`train_model.py`** like the example above in files explanation section. Now, we should have .model or .h5, label encoder file ending with .pickle, and image in the output folder you specify. If you follow exact code above in files explanation, you should see liveness.model or liveness.h5, label_encoder.pickle, and plot.png in this exact folder (like in this repo).
8. Run **`liveness_app.py`** like the example above in files explanation section and see whether it works well. If the model always misclassify, go back and see whether you save output images (real/fake) in the right folder. If you are sure that you save everything in the right place, collect more data or better quality data. *This is the common iterative process of training model, don't feel bad if you have this problem.*

## What can be improved
- Collect more data in many light conditions and from different genders/ethnics to improve the model (it turned out light intensity and condition play a big role here)

## Resources
- https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
- https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/
- https://www.youtube.com/watch?v=2Zz97NVbH0U&t=790s
### Source
- https://github.com/jomariya23156/face-recognition-with-liveness-web-login