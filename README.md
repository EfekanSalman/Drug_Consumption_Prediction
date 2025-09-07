# Drug Consumption Prediction Project

A comprehensive machine learning project that predicts drug consumption patterns using personality traits and demographic information. This project implements a robust, production-ready pipeline with experimental tracking, comprehensive testing, and modular architecture.

## 🎯 Project Overview

This project explores the potential of predicting drug consumption using personality traits and demographic information based on a dataset obtained from Kaggle. The workflow includes data preprocessing, machine learning modeling, hyperparameter tuning, and comprehensive model evaluation with MLflow integration.

## 🏗️ Project Structure

```
Drug_Consumption_Prediction/
├── data/                          # Data directory
│   ├── Drug_Consumption.csv      # Main dataset
│   └── raw/                      # Raw data backup
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py                   # Main entry point with CLI
│   ├── preprocessing.py          # Data preprocessing transformers
│   ├── train.py                  # Training workflow with MLflow
│   ├── inference.py              # Model inference and prediction
│   └── utils.py                  # Utility functions
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── conftest.py              # Test fixtures
│   ├── test_preprocessing.py     # Preprocessing tests
│   └── test_utils.py            # Utility function tests
├── notebooks/                    # Jupyter notebooks for EDA
├── requirements.txt              # Python dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                     # This file
```

## 🚀 Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for preprocessing, training, and inference
- **MLflow Integration**: Comprehensive experimental tracking with automatic logging of hyperparameters, metrics, and model artifacts
- **Comprehensive Testing**: Full test coverage with pytest including unit tests for all critical components
- **Production Ready**: Robust error handling, logging, and model persistence
- **CLI Interface**: Easy-to-use command-line interface for training and inference

## 📊 Key Findings

Our model successfully predicted cannabis use with **82% accuracy**. Feature importance analysis revealed that **psychological and personality scores** (Oscore, SS, Cscore, Nscore, AScore) were more influential in predicting drug consumption than demographic variables.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EfekanSalman/drug-consumption-prediction.git
   cd drug-consumption-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, sklearn, mlflow; print('All dependencies installed successfully!')"
   ```

## 🎮 Usage

### Command Line Interface

The project provides a flexible CLI for different operations:

#### Training Mode
```bash
# Train a new model with default settings
python src/main.py --mode train

# Train with custom logging level
python src/main.py --mode train --log-level DEBUG
```

#### Inference Mode
```bash
# Run inference on test data
python src/main.py --mode inference
```

### Programmatic Usage

#### Training
```python
from src.train import main as train_main

# Train a model
model, results, feature_importance = train_main()
```

#### Inference
```python
from src.inference import DrugConsumptionPredictor

# Load trained model and make predictions
predictor = DrugConsumptionPredictor()
predictions = predictor.predict(new_data)
```

#### Preprocessing
```python
from src.preprocessing import DrugDataPreprocessor, prepare_target_variable

# Preprocess data
preprocessor = DrugDataPreprocessor()
preprocessor.fit(training_data)
processed_data = preprocessor.transform(new_data)

# Prepare target variable
target = prepare_target_variable(data, 'Cannabis', is_binary=True)
```

## 🧪 Testing

The project includes comprehensive unit tests using pytest:

### Run All Tests
```bash
pytest
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test Files
```bash
pytest tests/test_preprocessing.py
pytest tests/test_utils.py
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Coverage Reports**: HTML coverage reports generated in `htmlcov/`

## 📈 MLflow Integration

The project uses MLflow for comprehensive experimental tracking:

### View Experiments
```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Tracked Information
- **Parameters**: Hyperparameters, dataset info, model configuration
- **Metrics**: Accuracy, F1-score, precision, recall, feature importance
- **Artifacts**: Trained models, plots, and logs
- **Metadata**: Run names, timestamps, and experiment organization

### Experiment Management
- Automatic experiment creation and organization
- Run comparison and model selection
- Model versioning and deployment tracking

## 📋 Data Preprocessing

The `DrugDataPreprocessor` class handles comprehensive data preprocessing:

### Features Processed
- **Gender**: Mapped to numerical values (F=0, M=1, unknown=-1)
- **Education**: Mapped to 1-9 scale based on education level
- **Categorical Variables**: One-hot encoded (Country, Ethnicity)
- **Drug Columns**: Removed from features (used as targets)
- **Irrelevant Columns**: Removed (ID, Age)

### Target Variable Preparation
- **Binary Classification**: CL0/CL1 (Low Risk) vs CL2+ (High Risk)
- **Multi-class Support**: Original consumption levels preserved

## 🔧 Configuration

### Hyperparameter Tuning
The training process includes comprehensive hyperparameter optimization:

```python
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
}
```

### Model Pipeline
- **Preprocessing**: Custom transformer for data cleaning
- **SMOTE**: Synthetic minority oversampling for class balance
- **Classifier**: Random Forest with hyperparameter tuning
- **Cross-validation**: 5-fold CV for robust evaluation

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **Feature Importance**: Top contributing features

### Model Artifacts
- Trained model saved as `cannabis_model.pkl`
- MLflow model registry for versioning
- Feature importance analysis
- Confusion matrix visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests before committing
pytest

# Check code coverage
pytest --cov=src --cov-report=term-missing
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle Drug Consumption Dataset](https://www.kaggle.com/datasets/...)
- MLflow for experimental tracking
- scikit-learn for machine learning algorithms
- imbalanced-learn for handling class imbalance

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation in the code

---

**Note**: This project is for educational and research purposes. The predictions should not be used for clinical or legal decision-making.