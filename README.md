Overview

This repository contains four machine learning projects demonstrating complete workflows of data preprocessing, feature engineering, model training, and evaluation. These tasks cover practical applications in text classification, image recognition, regression, and multi-class classification.

The tasks included are:

Email Spam Classification – classifying emails as spam or not spam using text data.

MNIST Digit Recognition – recognizing and classifying handwritten digits from 0 to 9 using image data.

Task 1: Email Spam Classification

Objective: Build a machine learning model to classify emails as spam or not spam using a labeled dataset.

Workflow:

Data Preprocessing: Cleaned and normalized raw email text; removed unnecessary characters, punctuation, and stopwords.

Feature Extraction: Converted text into numerical features using vectorization techniques.

Model Training: Trained a supervised learning model (e.g., Naive Bayes or Logistic Regression) on the processed dataset.

Evaluation: Evaluated the model using accuracy, precision, recall, and confusion matrix.

Files included:

spam_classification.ipynb – complete code for preprocessing, training, and evaluation.

Task 2: MNIST Digit Recognition

Objective: Build a machine learning model to recognize and classify handwritten digits (0–9) from the MNIST dataset.

Workflow:

Data Loading & Preprocessing: Loaded MNIST images in IDX format; normalized pixel values and reshaped images into feature vectors.

Model Training: Trained a Logistic Regression model to classify digits.

Evaluation: Measured performance using accuracy, classification report, and confusion matrix; visualized sample predictions.

Files included:

mnist_digit_recognition.ipynb – complete code for preprocessing, training, evaluation, and visualization.
