import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv")
obj_cols = train_data.select_dtypes(include="object").columns.to_numpy()
w = np.load("weights.npy")

def handleNAN(data: pd.DataFrame) -> None:

    # Numeric columns that can replace NAN entries with 0.
    zero_fill_cols = ["GarageYrBlt", "MasVnrArea", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFinSF1", "2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "Fireplaces", "GarageCars", 
                  "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]
    
    # Numeric columns that should replace NAN entries with the median of the neighborhood.
    median_fill_cols = ["LotFrontage", "LotArea", "MSSubClass", "OverallCond", "OverallQual", "YearRemodAdd", "YearBuilt", "GrLivArea", "FullBath", "HalfBath", "1stFlrSF", "BedroomAbvGr", 
                        "KitchenAbvGr", "TotRmsAbvGrd", "MoSold", "YrSold"]
    
    # Replace NAN entries for numeric columns.
    data[zero_fill_cols] = data[zero_fill_cols].fillna(0)
    data[median_fill_cols] = data[median_fill_cols].fillna(data[median_fill_cols].median())

    # For the other categorical columns, fill with "None".
    data.fillna("None", inplace=True)

def create_categorical_variables(data: pd.DataFrame, col: str, categories: list[str]) -> None:

    # Create a categorical variable for each category.
    for category in categories:
        new_col = col + "-" + category
        data[new_col] = data[col].apply(lambda x: 1 if category in str(x) else 0)
        
    # Drop the original column.
    data.drop(col, axis=1, inplace=True)

def createTestingDataFrame(testFilename: str) -> pd.DataFrame:
    test_data = pd.read_csv(testFilename)
    handleNAN(test_data)
    for col in obj_cols:
        categories = list(train_data[col].dropna().unique())

        if "None" not in categories:
            categories.append("None")
            
        create_categorical_variables(test_data, col, categories)
    
    return test_data

def getTesting(testFileName: str) -> tuple[np.ndarray, np.array]:
    test_data = createTestingDataFrame(testFileName)

    test_ids = test_data["Id"]

    X_test = test_data.drop(columns=["Id"])

    # Normalization:
    stats_df = pd.read_csv("norm_stats.csv")

    features = stats_df["feature"].tolist()
    mean = stats_df.set_index("feature")["mean"]
    std = stats_df.set_index("feature")["std"]

    X_test[features] = (X_test[features] - mean) / std

    X_test = X_test.values

    # Add bias column
    b_test = np.ones((X_test.shape[0], 1))

    X_test = np.concatenate([X_test, b_test], axis=1)

    return X_test, test_ids.to_numpy()

def predict(filename: str) -> None:
    X_test, test_ids = getTesting(filename)
    y = np.exp(np.dot(X_test, w))

    pred_df = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": y
    })

    pred_df.to_csv("predictions.csv", index=False)


predict("test.csv")
