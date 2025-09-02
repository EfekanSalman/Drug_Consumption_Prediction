# Drug Consumption Prediction Project

This project explores the potential of predicting drug consumption using personality traits and demographic information based on a dataset obtained from Kaggle. The workflow includes data preprocessing, machine learning modeling, and model performance evaluation.

## Project Structure
- **data/**: Contains the raw dataset (`Drug_Consumption.csv`).
- **src/**: Contains the Python source code.
  - **preprocessing.py**: Includes classes and functions for data cleaning and feature engineering.
  - **main.py**: Loads the dataset, applies preprocessing, trains a machine learning model, and evaluates its performance.

## Key Findings
Our initial model successfully predicted cannabis use with **82% accuracy**.  
Feature importance analysis revealed that **psychological and personality scores** (Oscore, SS, Cscore, Nscore, AScore) were more influential in predicting drug consumption than demographic variables.

## Requirements
All required libraries are listed in the `requirements.txt` file.

## How to Run
To run the project in your local environment, follow these steps:

1. Clone the project repository.
    ```bash
   pip install -r requirements.txt
    ```

2. Install the required libraries:
   ```bash
   git clone https://github.com/EfekanSalman/drug-consumption-prediction.git
    ```
   
3. Run the main script:
    ```bash
    python src/main.py
     ```