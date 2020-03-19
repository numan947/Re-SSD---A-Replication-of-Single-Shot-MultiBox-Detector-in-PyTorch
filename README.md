# Description

**I followed [1] for most of the coding, but tried to structure the whole project in a more meaningful way.**

The whole project can be divided into 6 parts.
1. Reading, preprocessing and loading the data
2. Model design and implementation; loss formulation and implementation
3. Training model with validation and hyper parameter tuning (hopefully :D)
4. Processing outputs of the model using Non-Maximum Suppression (NMS)
5. Evaluating the model using mAP (11pt & all pt) on testing data
6. Detecting objects in images, videos and live feeds.

# 1. Reading, preprocessing and loading the data
## Data
- PascalVOC: 2007, 2012
- TODO
# 2. Model building
- TODO

# 3. Training Model
- TODO
# 4. Processing Output
- TODO
# 5. Evaluating Model
- TODO
# 6. Detecting Objects
- TODO
## 11 Point Interpolation
{'aeroplane': 0.7884710431098938,
 'bicycle': 0.8282076716423035,
 'bird': 0.7344603538513184,
 'boat': 0.6800249218940735,
 'bottle': 0.4149485230445862,
 'bus': 0.8429154753684998,
 'car': 0.8458565473556519,
 'cat': 0.8873502612113953,
 'chair': 0.546992301940918,
 'cow': 0.8079142570495605,
 'diningtable': 0.7247970104217529,
 'dog': 0.8680041432380676,
 'horse': 0.8733161091804504,
 'motorbike': 0.81952303647995,
 'person': 0.7613175511360168,
 'pottedplant': 0.4676790237426758,
 'sheep': 0.7524116635322571,
 'sofa': 0.7591263651847839,
 'train': 0.8569323420524597,
 'tvmonitor': 0.7407298684120178}

Mean Average Precision (mAP): 0.750

## All Point Interpolation
{'aeroplane': 0.817555844783783,
 'bicycle': 0.8592820167541504,
 'bird': 0.756231427192688,
 'boat': 0.6991036534309387,
 'bottle': 0.4026220142841339,
 'bus': 0.8784138560295105,
 'car': 0.8762062191963196,
 'cat': 0.9171531200408936,
 'chair': 0.5493982434272766,
 'cow': 0.838248074054718,
 'diningtable': 0.7483301162719727,
 'dog': 0.9021250605583191,
 'horse': 0.9091637134552002,
 'motorbike': 0.8495721817016602,
 'person': 0.786540150642395,
 'pottedplant': 0.4621800482273102,
 'sheep': 0.7712143063545227,
 'sofa': 0.7827737927436829,
 'train': 0.887152910232544,
 'tvmonitor': 0.7635340094566345}

Mean Average Precision (mAP): 0.773

# References
1. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
2. https://github.com/rafaelpadilla/Object-Detection-Metrics
3. https://gist.github.com/jkjung-avt/605904dc05691e44a26bc57bb50d3f04