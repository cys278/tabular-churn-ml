import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix
)

PROCESSED_PATH="data/processed"

def load_splits():
    X_train=pd.read_csv(f"{PROCESSED_PATH}/X_train.csv")
    X_val= pd.read_csv(f"{PROCESSED_PATH}/X_val.csv")
    X_test = pd.read_csv(f"{PROCESSED_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_PATH}/y_train.csv")
    y_val = pd.read_csv(f"{PROCESSED_PATH}/y_val.csv")
    y_test = pd.read_csv(f"{PROCESSED_PATH}/y_test.csv")
    return X_train,X_val,X_test,y_train,y_val,y_test
