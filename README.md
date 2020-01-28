# Key Detection Test

A simple test script that can differentiate between two keys.

Two different keys—which are labeled key A and key B—are added to the training set. A seperate image of key A, with a different background, is added to the test set. This key is then correctly identified as key A.

I use OpenCV to seperate the keys from their background using [adaptive thresholding](https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html). I then run these threshold through OpenCV's [matchShape](https://www.learnopencv.com/shape-matching-using-hu-moments-c-python/) to identify the keys.

# Install

You will need to install python packages *python-opencv* and *imutils* to use this script.
I recommend using python's package manager pip.

```
pip3 install opencv-python
pip3 install imutils
```