UAE Rent Forecasting
====================

The UAE Rent Forecasting project is a comprehensive pipeline designed to predict rental prices across the UAE using advanced machine learning and time-series forecasting methods. This project integrates models like ARIMA, Prophet, LSTM, PyTorch, TensorFlow, Random Forest, and others to provide accurate predictions for decision-making in the real estate sector.

---

Project Structure
-----------------

1. **Data Directory**:
   - `data/raw/`: Unprocessed datasets (e.g., monthly rent prices).
   - `data/processed/`: Cleaned and feature-engineered datasets ready for modeling.
   - `data/forecasts/`: Generated forecast files.

2. **Models Directory**:
   - `models/`: Stores trained models, such as:
     - `prophet_model.pkl`: Trained Prophet model.
     - `lstm_model.h5`: TensorFlow-based LSTM model.
     - `random_forest_model.pkl`: Random Forest model.
     - `pytorch_model.pt`: PyTorch-based deep learning model.
     - `tensorflow_model.h5`: TensorFlow model.

3. **Notebooks Directory**:
   - `notebooks/`: Contains Jupyter notebooks for:
     - **eda.ipynb**: Exploratory Data Analysis.
     - **model_training.ipynb**: Step-by-step training and evaluation.
     - **forecasting.ipynb**: Forecast generation.

4. **Source Code Directory**:
   - `src/`: Core scripts for the pipeline:
     - `preprocess.py`: Cleans and structures raw datasets.
     - `train.py`: Implements model training routines.
     - `forecast.py`: Generates future predictions using trained models.
     - `evaluate.py`: Computes evaluation metrics (e.g., RMSE, MAPE).
     - `utils.py`: Includes helper functions like data loaders.

5. **Tests Directory**:
   - `tests/`: Contains unit tests for robustness:
     - `test_preprocessing.py`: Validates preprocessing logic.
     - `test_training.py`: Tests model training routines.
     - `test_forecasting.py`: Ensures accurate forecast generation.

6. **Other Files**:
   - `.gitignore`: Specifies files/folders to exclude from version control.
   - `LICENSE`: License for the project.
   - `README.txt`: This file.
   - `requirements.txt`: List of required Python libraries.
   - `main.py`: Command-line interface to orchestrate the pipeline.

---

How to Use
----------

1. **Setup**:
   - Clone the repository:
     ```
     git clone https://github.com/your-username/uae-rent-forecasting.git
     cd uae-rent-forecasting
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - (Optional) Use a virtual environment:
     ```
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

2. **Commands**:
   - Preprocess Data:
     ```
     python main.py preprocess --input data/raw/monthly_rent_data.csv --output data/processed/cleaned_data.csv
     ```
   - Train a Model:
     - Prophet:
       ```
       python main.py train --model prophet
       ```
     - LSTM:
       ```
       python main.py train --model lstm --epochs 50 --gpu
       ```
     - Random Forest:
       ```
       python main.py train --model random_forest
       ```
   - Generate Forecasts:
     ```
     python main.py forecast --model prophet --horizon 12 --output data/forecasts/forecast_results.csv
     ```
   - Evaluate Models:
     ```
     python main.py evaluate --model prophet --metrics rmse mae
     ```

---

Dependencies
------------

The `requirements.txt` file includes:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- torch
- tensorflow
- prophet
- joblib
- jupyter

Install dependencies with:
