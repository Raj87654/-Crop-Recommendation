import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Loads the crop recommendation dataset."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Separates features and target, and encodes the target variable."""
    X = df.drop('label', axis=1)
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
