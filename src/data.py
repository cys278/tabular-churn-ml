import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_DATA_PATH="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH="data/processed"

def load_data(path=RAW_DATA_PATH):
    """Load the Dataset"""
    df=pd.read_csv(path)
    return df

def split_and_save(df,target_col="Churn"):
    """Split into train/val/test and save to data/processed"""
    os.makedirs(PROCESSED_PATH,exist_ok=True)

    # Convert target column to binary (Yes=1,No=0)
    df[target_col]=df[target_col].map({"Yes":1,"No":0})

    # Seperate features and labels
    X= df.drop(columns=[target_col])
    y=df[target_col]

    # First split: train vs temp (val+ test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30,stratify=y, random_state=42
    )

    # Second split: validation vs test

    X_val, X_test, y_val, y_test=train_test_split(
        X_temp,y_temp, test_size=0.50,stratify=y_temp,random_state=42
    )

    # save to splits
    X_train.to_csv(f"{PROCESSED_PATH}/X_train.csv",index=False)
    X_val.to_csv(f"{PROCESSED_PATH}/X_val.csv",index=False)
    X_test.to_csv(f"{PROCESSED_PATH}/X_test.csv",index=False)

    y_train.to_csv(f"{PROCESSED_PATH}/y_train.csv", index=False)
    y_val.to_csv(f"{PROCESSED_PATH}/y_val.csv", index=False)
    y_test.to_csv(f"{PROCESSED_PATH}/y_test.csv", index=False)
    print("Data split and saved in data/processed/")


if __name__=="__main__":
    df=load_data()
    print("Raw Shape: ",df.shape)
    split_and_save(df)







