uae-rent-forecasting/
├── data/
│    ├── raw/                           # Raw datasets
│    ├── processed/                     # Cleaned data ready for modeling
│    └── forecasts/                     # Generated forecasts
├── models/
│    ├── prophet_model.pkl              # Saved Prophet model
│    ├── lstm_model.h5                  # Saved LSTM model
│    ├── random_forest_model.pkl        # Saved Random Forest model
│    ├── pytorch_model.pt               # Saved PyTorch model
│    └── tensorflow_model.h5            # Saved TensorFlow model
├── notebooks/
│    ├── eda.ipynb                      # Exploratory data analysis
│    ├── model_training.ipynb           # Model training and evaluation
│    └── forecasting.ipynb              # Generating forecasts
├── src/
│    ├── preprocess.py                  # Data preprocessing script
│    ├── train.py                       # Script for training models
│    ├── forecast.py                    # Script for generating forecasts
│    ├── evaluate.py                    # Evaluation metrics computation
│    └── utils.py                       # Utility functions (e.g., data loaders)
├── tests/
│    ├── test_preprocessing.py          # Unit tests for preprocessing
│    ├── test_training.py               # Unit tests for training logic
│    └── test_forecasting.py            # Unit tests for forecasting
├── docs/                               # Documentation for the project
├── animations/                         # Optional animations/plots for presentation
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── main.py                             # Main orchestrator CLI
