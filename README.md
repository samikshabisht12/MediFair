# Hospital Price Fairness and Anomaly Detection System

## Overview
The Hospital Price Fairness Analyzer is a machine learning-based system designed to evaluate and predict the fair cost of various medical services across different hospital locations. By leveraging historical hospital pricing data, the system predicts the fair price of a service and flags records that fall significantly above or below the expected range. It also features anomaly detection algorithms to help users identify deeply suspicious pricing behaviors.

## Features
- **Fair Price Prediction:** Utilizes a Linear Regression model trained on historical data (including metrics like wait times, hospital ratings, and cities) to estimate what a medical procedure should cost.
- **Fairness Classification:** Automatically categorizes billed prices into "Fair", "Overpriced", or "Underpriced" categories by comparing the actual price to the model's predicted price.
- **Anomaly Detection:** Incorporates an Isolation Forest algorithm to detect severe anomalies and outliers within the pricing datasets, effectively finding suspicious data points that may warrant further auditing.
- **Interactive Web Interface:** Provides an easy-to-use Streamlit web application that allows patients and auditors to input specific service details and instantly see the fairness assessment.

## Project Structure
- `app.py`: The Streamlit web application script for interactive usage.
- `model_training.py`: The data analysis and model training script. It handles data cleaning, feature encoding, training the Linear Regression and Isolation Forest models, and outputting data visualizations.
- `data/hospital_prices.csv`: The core dataset containing hospital, city, service type, pricing, wait time, and rating information.
- `price_model.pkl`: The serialized machine learning model used by the web interface.
- `requirements.txt`: The Python dependencies needed to run the project.

## Prerequisites
To run this project, you need Python installed on your system. It is highly recommended to use a virtual environment.

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Training the Model
If you have updated the dataset or wish to view the exploratory data analysis, run the training script. This will output data shapes, fairness evaluations, and scatter plots, and will save a new `price_model.pkl` file.
```bash
python model_training.py
```

### 2. Running the Web Application
To start the interactive interface and evaluate pricing fairness, run the Streamlit application:
```bash
streamlit run app.py
```
This will start a local web server and open the application in your default web browser. From there, you can select the medical service, city, input the hospital ratings, and compare the quoted actual price against the predicted fair price.

## Technical Details
The system utilizes one-hot encoding for categorical variables such as city and medical service type. The target variable for the regression model is the service price. For anomaly detection, the Isolation Forest considers both actual and predicted prices to flag highly irregular data points. Overpricing logic flags a service as "Overpriced" if the actual price exceeds the predicted price by more than 20 percent.
