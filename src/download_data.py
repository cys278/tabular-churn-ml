import kagglehub
import os, shutil

def download_telco_churn():
    # Download the dataset (from KaggleHub cache)
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("Downloaded dataset to cache:", path)

    # Find the CSV inside
    csv_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    src = os.path.join(path, csv_name)

    # Destination (data/raw/)
    dst = os.path.join("data", "raw", csv_name)
    os.makedirs("data/raw", exist_ok=True)

    # Copy only if not already there
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Dataset copied to {dst}")
    else:
        print(f"Dataset already exists at {dst}")

    return dst

if __name__ == "__main__":
    download_telco_churn()
