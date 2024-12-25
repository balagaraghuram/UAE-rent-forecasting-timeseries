import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    """
    Preprocess raw time-series data.
    - Handles missing values
    - Adds lag features or rolling statistics
    - Outputs cleaned data
    """
    df = pd.read_csv(input_path, parse_dates=['date'], index_col='date')
    
    # Example preprocessing: Fill missing values with median
    df = df.fillna(df.median())
    
    # Add rolling mean and standard deviation
    df['rolling_mean'] = df['rent_price'].rolling(window=3).mean()
    df['rolling_std'] = df['rent_price'].rolling(window=3).std()
    
    # Drop rows with NaN values created during rolling computations
    df.dropna(inplace=True)
    
    df.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")
    return df
