# Treue_Technologies-Task_4-Email_Spam_Detection

    This repository contains code for detecting email spam using machine-learning techniques. The code is implemented in Python and utilizes libraries such as pandas, NumPy, matplotlib, seaborn, sci-kit-learn, and imbalanced-learn.

INTRODUCTION:
    
    Email spam detection is a critical task to identify and filter out unwanted emails that could contain malicious content or unsolicited advertisements. In this project, we analyze an email dataset, preprocess the data, apply various machine learning models, and evaluate their performance for spam detection.

IMPORTING LIBRARIES:

    Importing the libraries are numpy, pandas, matplotlib, seaborn, Logistic Regression, Random Forest, Decision Tree and SVC classifiers
    
Contents:

    Import necessary libraries for data manipulation, visualization, and warnings handling.
    Load the email dataset from a CSV file and explore its contents.
    Preprocess the data by handling missing values and removing unnecessary columns.
    Visualize the distribution of spam and non-spam emails using a countplot.
    Label encode the target column and perform feature extraction using TF-IDF vectorization.
    Split the data into training and testing sets.
    Address data imbalance using oversampling techniques.
    Build and train different classification models: Logistic Regression, Support Vector Classifier, Random Forest, and Decision Tree.
    Evaluate the models' performance using confusion matrices, accuracy scores, and classification reports.
    Use the trained models to predict whether given emails are spam or not.

Results:

    Data Loading and Exploration: Load the email dataset and explore its contents.
    Data Preprocessing: Clean the data, handle missing values, and remove unnecessary columns.
    Data Visualization: Visualize the distribution of spam and non-spam emails using a countplot.
    Feature Extraction: Convert email text into numerical features using TF-IDF vectorization.
    Data Splitting: Split the data into training and testing sets.
    Data Balancing: Address data imbalance using oversampling techniques.
    Model Building: Train different classification models (Logistic Regression, SVM, Random Forest, Decision Tree).
    Model Evaluation: Evaluate model performance using confusion matrices, accuracy scores, and classification reports.
    Model Prediction: Use trained models to predict whether given emails are spam or not.
    
CONCLUSION:

            The above msg all models have been able to tell that the message is spam
