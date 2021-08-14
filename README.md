# VFX Group Project --- Real-time Gesture Detection and Animation
### Team Members: Yue SHEN, Santiago Gomez MAQUEO & Emilie HEBRARD
### Organization: AI Launch Lab
<br />

## Project Description
### Overview:
- This project aims to build AI tools for gesture-based interfaces using indigenous perspectives.
- This group's main focus is to generate animations and real-time detection of gestures.
### Problem types:
- Supervised learning, Computer Vision, Classification (Machine Learning Task).
### Datasets:
 - Kaggle gesture dataset: https://www.kaggle.com/gti-upm/leapgestrecog
 - This dataset is composed by a set of near infrared images acquired by the Leap Motion sensor.
 - The database is composed by 10 different hand-gestures that were performed by 10 different subjects (5 men and 5 women).
### Protocols we follow:
 - Indigenous Protocol and AI, 2020: https://spectrum.library.concordia.ca/986506/7/Indigenous_Protocol_and_AI_2020.pdf
 <br />
 
 ## Deliverables
 - Final model trained using 2D CNNs based on VGG-16 architecture, in final_model_vgg16.ipynb
 - Implementation of real-time webcam gesture recognition with audio features and potential extended application in smart-home devices, in real_time_recog.py
 - Front-end implementation of gesture animations using Python Flask framework, in app.py
<br />

## Limitations and Progress
### Limited size and diversity of dataset:
  - **Consequences:** Reduced the accuracy of model prediction, even though cross validation and train-test split are applied;
    made it nearly impossible to predict colorful gesture images that are close to our everyday lives.
  - **Progress:** Used OpenCV (background masking and binary image thresholding) to preprocess images into binary black-and -white images, then sent into the model;
    transformed the image dataset into binary images, then trained the model using this newly-generated dataset; 
    (tried to use data augmentation but found of limited value in the improvement of model accuracy, because original dataset already inherited properties such as ramdom flip and     random rotation).
  - **Effects:** Model accuracy has been much improved, is able to predict colorful and daily gesture images but would be improved more if training dataset can be colorful and         random with higher diversity.
### Model selection:
  - **Initial Trial:** CNNs has been proved very suitable for training on image dataset, as such initially standard Keras layers such as Dense, Flatten were randomly added to         build the model with all activation functions being ReLU.
  - **Weakness:** However, a large number of hyper-parameters were used, leading to the overfitting of model, and could hardly predict new images even though they are black-and-       white.
  - **Progress:** Changed the model into a pre-trained model based on VGG-16 architecture, which is well-known for lightening the number of parameters and overcoming overfitting       issues; results were improved by applying this.
### Front-end applications:
   - Due to the lack of physical smart-home devices such as Philips Hue and low computations of PCs, it's hard to extend our animations and explore more on visual effects that          we initially would like to implement.
   - We are very willing to continue our research and develop further on optimizing this project!
