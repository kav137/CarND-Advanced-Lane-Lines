## Advanced Lane Finding Project

---



The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibrator1.png]: ./writeup_materials/calibrator1.png "Calibrator 1"
[calibrator2.png]: ./writeup_materials/calibrator2.png "Calibrator 2"
[calibrator3.png]: ./writeup_materials/calibrator3.png "Calibrator 3"
[calibrator7.png]: ./writeup_materials/calibrator7.png "Calibrator 7"
[calibrator11.png]: ./writeup_materials/calibrator11.png "Calibrator 11"
[calibrator12.png]: ./writeup_materials/calibrator12.png "Calibrator 12"
[calibrator13.png]: ./writeup_materials/calibrator13.png "Calibrator 13"

[undistorted1.png]: ./writeup_materials/undistorted1.png "Undistorted 1"
[undistorted2.png]: ./writeup_materials/undistorted2.png "Undistorted 2"
[undistorted3.png]: ./writeup_materials/undistorted3.png "Undistorted 3"

[combined_mask1.jpg]: ./writeup_materials/combined_mask1.jpg "Combined mask 1"
[combined_mask2.jpg]: ./writeup_materials/combined_mask2.jpg "Combined mask 2"
[combined_mask3.jpg]: ./writeup_materials/combined_mask3.jpg "Combined mask 3"
[combined_mask4.jpg]: ./writeup_materials/combined_mask4.jpg "Combined mask 5"
[mask.jpg]: ./writeup_materials/mask.jpg "Combined mask (white)"

[birdView1.png]: ./writeup_materials/birdView1.png "birdView 1"
[birdView2.png]: ./writeup_materials/birdView2.png "birdView 2"
[birdView3.png]: ./writeup_materials/birdView3.png "birdView 3"
[birdView4.png]: ./writeup_materials/birdView4.png "birdView 4"
[birdView5.png]: ./writeup_materials/birdView5.png "birdView 5"

[lineFinder.png]: ./writeup_materials/lineFinder.png "Line Finder"

[result1.jpg]: ./writeup_materials/result1.jpg "result 1"
[result2.jpg]: ./writeup_materials/result2.jpg "result 2"
[result3.jpg]: ./writeup_materials/result3.jpg "result 3"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

## Table of contents
* [Providing a writeup for the project](#providing-a-writeup-for-the-project)
* [Camera calibration](#camera-calibration)
* [Pipeline for test images](#pipeline-for-test-images)
  * [Distortion correction](#distortion-correction)
  * [Creating binary mask](#creating-binary-mask)
  * [Performing perspective transformations](#performing-perspective-transformations)
  * [Detecting lines and fit their position with polynominal](#detecting-lines-and-fit-their-position-with-polynominal)
  * [Calculating the radius of curvature](#calculating-the-radius-of-curvature)
  * [Plotting annotaions onto source images](#plotting-annotaions-onto-source-images)
* [Pipeline for video](#pipeline-for-video)
* [Discussion](#discussion)
---
## Providing a writeup for the project

You're currently reading it! Hope you'll enjoy =)

## Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In order to provide a convenient way of working with source code the code have been splitted in modules placed in ./src folder.
The code which covers calculations requried for distortion correction topics is placed within [calibrator](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/calibrator.py) file. Calibrator is a class with a set
of static methods and properties. Such approach allows to perform operations without craeting class instance, static
properties allow to store informations that should be calculated only once (for example imgpoints, objpoints,
info about image size and so on).

There are 2 methods within class which are used from the outside:
* [calibrate()](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/calibrator.py#L110)

   First of all method reads all the images in `camera_cal` folder. After that an attempt to find corners is made using
   `cv2.findChessboardCorners` method. If corners were found `imgpoins` and `objpoints` arrays are extended with corresponding values.
   When all the images were processed method calculates camera distortion coefficients, transformation matrix and ROI values.
   In order to get them `cv2.calibrateCamera` and `cv2.getOptimalNewCameraMatrix` methods are used.

* [undistort_image(image)](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/calibrator.py#L76)

   This method is used to create undistorted image using coefficients calculated earlier.

Here are some examples of how the module works for calibration images:

![Calibration example][calibrator1.png]
![Calibration example][calibrator2.png]
![Calibration example][calibrator3.png]
![Calibration example][calibrator7.png]
![Calibration example][calibrator11.png]
![Calibration example][calibrator12.png]
![Calibration example][calibrator13.png]

## Pipeline for test images

### Distortion correction.

Using method `Calibrator.undistort_image(image)` which was mentioned in the previous section it is possible to produce image which is not affected by the camera lenses distortion effect.

The easiest way to see how image distortion affects image is pay attention to the way the shape of the car's hood differs in the corners of the images. Take a look:

![Undistorted image][undistorted1.png]
![Undistorted image][undistorted2.png]
![Undistorted image][undistorted3.png]

### Creating binary mask

In order to detect lane lines on the image and highlight them I've used multipe filters:
* Sobel filtering by *x* and *y* directions ([code](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/edges.py#L7))

* Sobel magnitude filtering ([code](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/edges.py#L29))

* Angular edges filtering ([code](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/edges.py#L47))

* Thresholding *S* and *H* components in HSL image ([code](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/edges.py#L77))

First three approaches allow to detect lines effectively based on the direction lane lines are placed on the image, the
last one allows to filter yellow and white line effectively - this is especially important when there is very small
difference in contrast value on grey image.

Following values were used for thresholding:

Filter | Low | High | Kernel size
--- | --- | --- | ---
Sobel X | 50 | 150 | 15
Sobel Y | 50 | 150 | 15
Sobel magnitude | 50 | 150 | 9
Angular | 0.8 | 1.2 | 9
HLS (H component) | 15 | 35 | -
HLS (S component) | 150 | 230 | -

In order to combine all the filters described following grouping have been chosen:
```
    # 'hls' filter is [(hls_s == 1 & hls_h == 1)]
    combined_mask[((sobelx == 1) & (sobely == 1) & (angular == 1)) | ((sobelxy == 1) | (hls == 1))] = 1
```

Here are examples showing how each filter works in different conditions.
Color code:
* *Red* ~ (sobelx == 1) & (sobely == 1) & (angular == 1)
* *Green* ~ (hls == 1)
* *Blue* ~ (sobelxy == 1)

![Combined mask][combined_mask1.jpg]
![Combined mask][combined_mask2.jpg]
![Combined mask][combined_mask3.jpg]
![Combined mask][combined_mask4.jpg]

And here's how combined mask looks like (without channels separation):


![Mask][mask.jpg]

### Performing perspective transformations

The code which implements transform is placed in [birdView](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/birdView.py)
module which's public API is the single method of static class [`transform_perspective`](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/birdView.py#L9)

During the first time of methods invokation two matrices (straight and reverted) are calculated and cached
for further usage. Calculations are made using `cv2.getPerspectiveTransform` method.

There is no guarantee that camera is placed right at the centre of the car so we can't assume position of
the road (as the src points) using strategy like "take image center along *x* axis and calculate some margin".
So I've augmented image and got the exact coordinates using Illustrator.

So, the coordinates I've used:
```
    src_coords = np.float32([(596, 460), (713, 460), (1019, 666), (321, 666)])
    dest_coords = np.float32([(250, 0), (1030, 0), (1030, 720), (250, 720)])
```

And here are examples of converted images (first two were augmented manually):

![Bird View][birdView1.png]
![Bird View][birdView2.png]
![Bird View][birdView3.png]
![Bird View][birdView4.png]
![Bird View][birdView5.png]


### Detecting lines and fit their position with polynominal

In order to detect lines and fit them with polynominal I've used the code that was provided in the ND's lesson.
I've refactored it and put in the [lineFinder](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/lineFinder.py) module.
There are two methods to find_lines ([find_initial_lines](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/lineFinder.py#L21) and [find_next_lines](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/lineFinder.py#L98)) each which are used for cases when we either have infromation about previously detected lines or not.

The algorithm implemented within both methods is based on the same principle: we split the frame by *Y* axis
and scan for mask's pixels distribution within segment. The highest peaks are initially chosen as lines.
If during next scans we'll find some place that has more pixels than predefined threshold (I've increased it to the value 70 in order to reduce noizy line movements) assumed position of line is shifted there.

Fitting is made via `np.polyfit` method. The polynominal of the magnitude of 2 is used.
Image augmentations are made using `cv2.fillPoly` and `cv2.rectangle`

Here is an example of how finding process looks like:

![Line finder][lineFinder.png]

### Calculating the radius of curvature

Calculation of radius curvature is made using the code provided in the lesson and in general
are based on the Polynominal equation. Calculations are made with respect to
the scale (meters per visible pixel) so the values of curvature are fit the information that radius is approximately
1000 meters. The results I've got are oscillating in range between 600 and 1500 meters which are seemed to be pretty good.

All the code could be found in the separate [Measurements](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/measurements.py) module.

The same module includes a [method](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/measurements.py#L38) for measuring car's shifting respecting the center of the lane.

### Plotting annotaions onto source images

In order to annotate source image several steps should be done:
* Augmented with lane's projection image should be warped back to the initial perspective
* Mask should be filled with a polygon in order to highligt the lane
* Text annotations should be placed
* Augmentations and source image should be mixed together

All the code that is responsible for these actions is located in [Utils](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/src/utils.py) module.

During the perspective transformation [reverse matrix](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/c23194956b1c8deebe23f923135e1d3e6dca81f8/src/birdView.py#L6)
that was calculated earlier is used.

As a main mean of creating augmentations openCV methods were used. They are: `cv2.fillPoly`, `cv2.warpPerspective`,
`cv2.addWeighted`, `cv2.putText`.

Here are examples of augmented images:

![Result 1][result1.jpg]
![Result 2][result2.jpg]
![Result 3][result3.jpg]
### Pipeline for video

Here's a link to my video result: [watch](https://drive.google.com/open?id=1Anv-pqqW6UZCUZCptVrimllOOlPQWoEu) or [download](https://github.com/kav137/CarND-Advanced-Lane-Lines/blob/master/result_video.mp4)

Pipeline used for video processing is based on the "per frame" video processing, hence it is just a sequence of
operations described before.

Resulting code (the same could be found in `pipeline` notebook):
```
left_line = None
right_line = None

def pipeline(image):
    undistorted_image = Calibrator.undistort_image(image)
    transformed_image = BirdView.transform_perspective(undistorted_image)
    mask = get_mask(transformed_image)

    global left_line, right_line

    (left_line, right_line, left_fitx, right_fitx, ploty, result_img) = LineFinder.find_lines(
        mask,
        left_line,
        right_line
    )

    left_curvature, right_curvature = Measurements.get_curvature(ploty, left_fitx, right_fitx)
    shift = Measurements.get_shift(left_line, right_line)

    annotated_image = show_info(
        undistorted_image, mask, left_fitx, right_fitx, ploty, left_curvature, right_curvature, shift
    )

    return annotated_image
```
---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

During implementation I've faced multiple problems: first of all the code could be easily led to the situation
when the lane is lost and the logic wouldn't try to reset its state in order to start new cycle of search -
it would stuck trying to base assumptions on the previous (non-existing) result. I think that more sophisticated
approach should be used and the system should have a dynamic level of confidence: whether the lane it works with is
a real one. Another problem is lane detection: current pipeline works only for particular whether/lightning conditions
which is of course is not good, because we can't assume driving only when its sunny outside. Guess, that image
processing pypeline should be more robust and parameters for filters should be calculated dynamically based on the
w/b balance of image, contrast, color levels and so on.
