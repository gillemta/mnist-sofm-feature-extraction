# Neural Network & SOFM for MNIST Classification

This repository contains the implementation for Problem 2 of Homework 5 from the Intelligent Systems class (EECE 5136). It implements a Self-Organizing Feature Map (SOFM) to extract and visualize features from the MNIST dataset. Although this code represents Problem 2, the accompanying report covers all three problems from Homework 5.

## Overview

This project implements a Self-Organizing Feature Map (SOFM) trained on a balanced subset of the MNIST dataset. The SOFM learns a 12×12 map where each neuron has 784 inputs corresponding to the 28×28 pixel images. After training, the model:
- Generates activity matrices for each digit (0-9) by identifying winning neurons.
- Visualizes the neuron weights as 28×28 images arranged in a 12×12 grid.

This approach demonstrates unsupervised feature extraction and provides insight into how the network organizes high-dimensional data.

## Features

- **Unsupervised Learning:** Implements a Self-Organizing Feature Map without relying on external libraries for the algorithm.
- **Activity Matrices:** Computes and visualizes activity matrices (heatmaps) for each digit class, showing the winning fraction of neurons.
- **Weight Visualization:** Rearranges the 784 weights of each neuron into 28×28 images to illustrate the features each neuron has learned.
- **Data Preprocessing:** Normalizes and reshapes the MNIST data for effective training.

## Installation & Setup

1. **Clone the Repository:**
  ```bash
  git clone https://github.com/gillemta/mnist-sofm-feature-extraction.git
  cd mnist-sofm-feature-extraction
  ```

2. **Create and Activate a Virtual Environment:**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

3. **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

4. **Run the Application:**
  ```bash
  python main.py
  ```

---

## Usage

To run the SOFM implementation and generate visualizations:

1. **Run the Application**

2. **What to Expect**
- The program will normalize and reshape the MNIST data.
- It trains the SOFM for a defined number of epochs
- After training, it will display:
  - Heatmaps for each digit's activity matrix.
  - A composite image showing the neuron weights arranged as a 12x12 grid of 28x28 images

---

## Results & Analysis

During training, the SOFM learns to organize the high-dimensional MNIST data into a 12x12 grid. The resulting activity matrices provide insight into which neurons are most active for each digit class. The weight visualization shows  the features each neuron has become sensitive to, effectively mapping the input space of handwritten digits.

For detailed analysis and a comparison with the other problems from Homework 5, please refer to the final report.

---

## Report

A comprehensive report detailing:
- The system design and hyperparameter choices.
- Experimental results including confusion matrices, error fractions, and performance comparisons.
- In-depth analysis of the classifier performance and feature visualizations.

The full report is available in the `/docs` folder or can be downloaded directly from [Final Homework 5 Report](link-to-report.pdf).
