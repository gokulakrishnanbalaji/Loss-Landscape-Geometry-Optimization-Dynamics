Name: Gokulakrishnan B
Roll No: D24M007
Course: M.Tech Data Science and AI

**Make sure to check the report, I have written detailed explanation on my apporach and observed results**

## Problem Statement 3

Loss Landscape Geometry &amp; Optimization Dynamics
Develop a rigorous framework for analyzing neural network loss landscape
geometry and its relationship to optimization dynamics, generalization, and architecture
design. Derive theoretical results, implement efficient landscape probing methods, and
empirically validate connections between geometric properties and model behavior.
Technical Background
Neural network optimization remains poorly understood:

* Why does SGD find generalizable minima despite non-convexity?

* How does architecture affect loss landscape topology?

* What geometric properties correlate with trainability and generalization?

* Can we predict optimization difficulty from landscape analysis?

Challenge: Develop methods to efficiently characterize loss landscapes and establish
rigorous connections to optimization outcomes.

## My approach
For detailed explanation, kindly refer the report, I'll explain briefly here.

I chose 3 models (MLP, CNN with and without residual), trained for 3 epochs on MNIST dataset. Then I used 3 approaches to understand the loss structure and one approach to optimise using **mode connectivity**. You can see the images of each approach attached as png file in the repo. I have shared my observation from the images in the report.
