# Facilitating Object Detection with Gaze Information for Kitchen Environments
![sample](https://user-images.githubusercontent.com/26382762/37161222-5ffc149a-22f2-11e8-9d57-fa2b135e2325.png)
https://github.com/YandanYang/YOLO-with-Gaze-Information/demo.avi
![Demo Video](https://github.com/YandanYang/YOLO-with-Gaze-Information/raw/master/demo.avi) is uploaded as 'demo.avi'

(Because of the large size, here I didn't upload backup/*weights, or dataset.)

## 1 Getting Started

Here we combine [Pupil](https://docs.pupil-labs.com) and [Darknet](https://pjreddie.com/darknet/yolo/).

### 1.1 Prerequisites

- The pupil version(pupil-0.9.14) need python3. So I install python3.5 on Ubuntu 16.04.
- CUDA. We test with Cuda V9.0.176.
- Opencv. We test with oepncv-3.2.0-dev. This is also a dependency of Pupil. So we will install opencv later.

### 1.2 Installing

#### Pupil
The requirements for the installation of the pupil framework are listed in its [website](https://docs.pupil-labs.com). For linux user, you can go straight [here](https://docs.pupil-labs.com/#linux-dependencies).
The source code is also uploaded here as `pupil-0.9.14.zip`.
When installing opencv during this step, change :
```
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_TBB=ON -DWITH_CUDA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON ..
```
into
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_TBB=ON -D WITH_TBB=ON -D WITH_CUDA=OFF -D PYTHON2_NUMPY_INCLUDE_DIRS=' WITH_QT=ON -D WITH_GTK=ON -D ..
```
This is because darknet will need opencv build with gtk.


#### Darknet
![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

We have made some changes to original darknet, so you may just download it here.
- Download this branch.
- Unzip it and rename it to be darknet. I put this fold in `~/otherSw/`. Some of the path in the code is writen according to this.
- Install darknet on your computer with the following step:

I have already modify the Makefile as:
```
GPU=1
CUDNN=0
OPENCV=1
```
If you could use CUDNN, you can also add it.

Then you need only two steps:
```
cd ~/otherSw/darknet  #go to the path you set
make
```

#### Demo recording of pupil
Move folder `recordings` into home path.
```
cd ~/otherSw/darknet
cp -r recordings ~/
```

Now you have downloaded all you need and everything installed.

## 2 Simple test
You should modify `~/otherSw/darknet/cfg/yolo41_kitchen.cfg` for testing instead of training.
`~/otherSw/darknet/cfg/yolo41_kitchen.cfg` should look like this:
```
 [net]
 # Testing
  batch=1
  subdivisions=1
 # Training
 # batch=16
 # subdivisions=8
 ....
```

```
cd ~/otherSw/darknet
```
Since we only make change to demo module, so the third argument is `demo`. Details for other parameters, you can refer to [this website](https://pjreddie.com/darknet/yolo/).

Run:
```
./darknet detector demo ~/otherSw/darknet/cfg/kitchen.data ~/otherSw/darknet/cfg/yolo41_kitchen.cfg ~/otherSw/darknet/backup/yolo41_kitchen_48000.weights /home/gazetracker/recordings/2018_02_17/006_95%/world.mp4
```
Then you will see a video as output, with single cross and bounding box on it.

## 3 Develop
Please remember, every time you modify `.c` or `.h` files, you need to compile darknet again.
### 3.1 Training Darknet
This training process has nothing to do with pupil. 
#### check .cfg
Modify `~/otherSw/darknet/cfg/yolo41_kitchen.cfg` for training instead of testing.
`~/otherSw/darknet/cfg/yolo41_kitchen.cfg` should look like this:
```
 [net]
 # Testing
 # batch=1
 # subdivisions=1
 # Training
 batch=16
 subdivisions=8
 ....
```
You can change the value of `batch` and `subdivisions` to fit it with your GPU memory and increase the training speed. 
Here the size and number of anchor-box is decided by k-means, written in `init_anchor_box.py`.
```
 [region]
 anchors = 1.80856, 2.43484 , 4.28920, 6.24500 , 12.83713, 7.54301 , 7.35418 , 13.89931, 16.32208 , 16.81458
```

#### dataset
We collected and labeled kitchen dataset(30,000+ images, 70,000+ labels), saved in  `~/otherSw/darknet/data/kitchen`. 
The data comes from COCO,VOC,Imagenet, and our lab's kitchen.
`transfer_coco2014.py` converts label from COCO2014, provided by YOLO author. And `rm_person.py` removes images that only have label for person.
`voc_label_yyd.py` convert label from [VOC dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/).
`imagenet_label_yyd.py` convert rare labels from imagenet to fit YOLO.

#### check .data
```
gedit ~/otherSw/darknet/cfg/kitchen.data
```
Then you will see:
```
classes= 41
train  = /home/gazetracker/otherSw/darknet/data/kitchen/train.txt
valid  = /home/gazetracker/otherSw/darknet/data/kitchen/val.txt 
names = data/kitchen.names
backup = backup/
EVAL = kitchen
```
`train.txt` and `val.txt` contains the whole kitchen dataset.

#### change frequency of saving weight
In `detector.c`, command `i%300==0` in line 136 decide the frequency to save weight. If you just want to see the final result, you can change it bigger, such as 10000.
Don't forget to run 
```
make clean
make
```
when you make any change to the code.

#### Training
Goto darknet path.
If you want to start with the weight given by YOLO author, run:
```
./darknet detector train cfg/kitchen.data cfg/yolo41_kitchen.cfg darknet19_448.conv.23 
```
If you want to start with our trained weight(might be overfitted), run:
```
./darknet detector train cfg/kitchen.data cfg/yolo41_kitchen.cfg backup/yolo41_kitchen_48000.weights 
```
If you want to train tiny-yolo model, run:
```
./darknet detector train cfg/kitchen.data cfg/tiny-yolo_kitchen.cfg darknet19_448.conv.23
```


### 3.2 Use Pupil to record 
```
cd ~/pupil/pupil_v0.9.14-7_linux_x64/pupil/src
python3 main.py
```
Then you can calibrate and record. The default path for records is ~/recordings. Please refer to [Pupil](https://docs.pupil-labs.com) for details. 
Do calibration every time before you record. Flip the eye image if it is upside-down.

### 3.3 Extract gaze position from Pupil
 The default saving path is `~/recordings`. Each recording of pupil will be saved in a folder. The folder contains original world video and timestamp for world camera. For gaze data, there are files recording coordinate and its timestamps seperately. Other recoding files also exists, while not used here.

Code in `darknet/python/extract_gaze_from_pupil.py` generates gaze position for each frame, saved in `pos_frame.txt`. You should change the  path in this code to fit your pupil recordings.

However, this code works only for off-line recordings. 

### 3.4 Apply gaze position into darknet
When we are testing darknet, the code will first go to `detecter.c`.

If we give video or webcam(ignore the last agrv) as an input, the code will then go to `demo.c`, where we introduced a function called `demo_pose`. In this function, we write the file name of gaze position, `pos_frame.txt`, and read the gaze data. But this is inconvenient, because we need to change the path if we want to test different video with different gaze position. A simple idea is to add the path as another agrv in command.

Also, it doesn't detect the end of txt file so we set the length at the beginning. But sorry for my poor coding skills, I failed. So the code for reading gaze data need to be improved. Function `demo_pose` reads the video frame by frame and count the number of processed frame. It will send the gaze position to the function `draw_detections_pose` in `image.c`. File `image.c` is to display image with bounding boxed and labels. `draw_detections_pose` is where I apply score algorithm. These two functions are included in appendix. Since I introduce some new-named functions, I also change the head files, `image.h` and `darknet.h`.
One more unfixed task is to save the output into video file. There are some guys sharing their codes online, using `CvVideoWriter`. But still, I got error when compiling it.
## 4 Deployment

The files in my computer is deployed as:
- ~/otherSw/darknet
- ~/pupil/pupil_v0.9.14-7_linux_x64/pupil
- ~/recordings 

## Built With

* [Pupil](https://docs.pupil-labs.com) 
* [Darknet](https://pjreddie.com/darknet/yolo/).

    For more information see the [Darknet project website](http://pjreddie.com/darknet).

    For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

## Cite
```
@article{redmon2016yolo9000,
  title={YOLO9000: Better, Faster, Stronger},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1612.08242},
  year={2016}
}
```
```
@inproceedings{Kassner:2014:POS:2638728.2641695,
 author = {Kassner, Moritz and Patera, William and Bulling, Andreas},
 title = {Pupil: An Open Source Platform for Pervasive Eye Tracking and Mobile Gaze-based Interaction},
 booktitle = {Adjunct Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
 series = {UbiComp '14 Adjunct},
 year = {2014},
 isbn = {978-1-4503-3047-3},
 location = {Seattle, Washington},
 pages = {1151--1160},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/2638728.2641695},
 doi = {10.1145/2638728.2641695},
 acmid = {2641695},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {eye movement, gaze-based interaction, mobile eye tracking, wearable computing},
}
```



