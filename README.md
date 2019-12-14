# EC504_Project
EC504 Project - Image Segmentation using Max Flow

# Overview
Image segmentation is a very common topic in computer vision in which an object is separated/segmented from the background. Generally, similar pixels are clustered together which helps to differentiate the foreground from the background. 

* In this project, a user uses an interactive command line interface (CLI) to draw lines on foreground and background. The foreground is an object the user wishes to segment out.

* All pixels in an image act as vertices of a graph and they are connected with their neighboring vertices using a weighted edge. The weights of the edges are determined by a boundary penalty formula and the log likelihood ratio of the probabilities given by the Gaussian Mixture Models. 

* Once the graph is set, Max-flow algorithm is used to create a min-cut between the foreground and background pixels. The area within the cut is the foreground which is segmented out and displayed on a white background.

* Edmonds-Karp algorithm, which is an implementation of Ford Fulkerson method, is used to segment out the object. Once graph-cut is performed, some image processing techniques are used for segmentation.

# How to Run

 - SCC

*module load opencv/4.0.1*

*module load python3/3.6.5*

* python main.py ./birdy.jpg

OR

* python mainGMM.py ./plane.jpg

## **File descriptions**
 - main.py runs max flow over graph with constant weights on terminal edges and boundary penalty weights on neighborhood edges
 - mainGMM.py runs max flow over graph with log likelihood ratio of GMM predicted probabilities weights on terminal edges and boundary penalty weights on neighborhood edges

## **Requirements**

*numpy 1.15.2*

*matplotlib 3.0.0*

*scikit-learn 0.20.0*

*python 3.6.5*

*opencv 3.4.2*

## **References**

* https://julie-jiang.github.io/image-segmentation/#algos
* Martin, David, et al. "A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics." Vancouver:: Iccv, 2001.
