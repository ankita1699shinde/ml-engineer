# ml-engineer
This project implements a deep learning pipeline to classify flower images by flower type, dominant color, and to estimate concentrations of three essential oils (simulated values) from flower images.
This code pipeline builds and trains a multi-task deep learning model to classify flower images by:

Type of Flower

Dominant Color

Estimate concentrations of 3 essential oils (simulated values)
Multi-Task Flower Classification & Oil Concentration Estimation
This project builds a deep learning model that can:

Classify the type of flower in an image (e.g., rose, tulip)

Identify the dominant color of the flower (red, white, green, yellow, blue, or other)

Estimate the concentrations of three essential oils (linalool, geraniol, citronellol) — these values are simulated

Project Overview
Dataset: Flower images organized by flower type

Labels:

Flower type (e.g., rose, tulip)

Dominant color

Simulated oil concentrations (linalool, geraniol, citronellol)

Model: Multi-task CNN using a pretrained ResNet18 backbone with separate heads for each task

Tasks:

Flower type classification (multi-class)

Dominant color classification (multi-class)

Oil concentration regression

Setup
Requirements
Python 3.8 or higher

PyTorch

torchvision

Pillow (PIL)

scikit-learn

numpy

pandas

Installation
Install all required packages using:

bash
Copy
Edit
pip install torch torchvision pillow scikit-learn numpy pandas
