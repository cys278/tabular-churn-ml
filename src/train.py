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

def train_and_eval():
    #Load data

    X_train,X_val,X_test,y_train,y_val,y_test=load_splits()

    #Identify numeric and categorical columns

    numeric_features=X_train.select_dtypes(include=["int64","float64"]).columns
    categorical_features=X_train.select_dtypes(include=["object","bool"]).columns

    # Preprocessing

    numeric_transformer=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scalar",StandardScaler())
    ])
    categorical_transformer=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor=ColumnTransformer(
        transformers=[
            ("num",numeric_transformer,numeric_features),
            ("cat",categorical_transformer,categorical_features)
        ]
    )

    #Model

    clf= Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=1000))
    ])

    # Train

    clf.fit(X_train,y_train)


    # Evaluate

    for split_name, X, y in [("Validation",X_val,y_val),("Test",X_test,y_test)]:
        y_pred=clf.predict(X)
        y_prob=clf.predict_proba(X)[:,1]
        print(f"\n{split_name} Metrics")
        print("AUC:",roc_auc_score(y,y_prob))
        print("F1:",f1_score(y,y_pred))
        print("Precision:",precision_score(y,y_pred))
        print("Recall: ",recall_score(y,y_pred))
        print("Confusion Matrix:\n",confusion_matrix(y,y_pred))

if __name__=="__main__":
    train_and_eval()






