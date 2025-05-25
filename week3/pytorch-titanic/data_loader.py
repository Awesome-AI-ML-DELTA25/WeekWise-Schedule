import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

torch.manual_seed(42)  # For reproducibility

def load_titanic_data(path):
    # Load dataset
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)

    # Target and features
    y = df['survived']
    X = df.drop('survived', axis=1)

    # Define column types
    categorical_cols = ['pclass', 'sex', 'embarked']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='drop'
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_train_tensor.shape[1]