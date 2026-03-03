import os
import joblib
import pandas as pd

from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    dataset_id = os.environ.get("DATASET_ID", "")
    local_path = os.environ.get("LOCAL_PATH", "")

    task = Task.init(project_name=project, task_name="TEMPLATE - preprocess", reuse_last_task_id=False)
    task.set_packages(requirements_file="requirements.txt")
    logger = task.get_logger()

    if dataset_id:
        raw_root = Dataset.get(dataset_id=dataset_id).get_local_copy()
    else:
        raw_root = local_path

    csv_path = os.path.join(raw_root, "breast_cancer.csv")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    out_dir = ensure_dir("data/processed")
    train_p = os.path.join(out_dir, "train.joblib")
    test_p = os.path.join(out_dir, "test.joblib")
    scaler_p = os.path.join(out_dir, "scaler.joblib")

    joblib.dump({"X_train": X_train, "y_train": y_train}, train_p)
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_p)
    joblib.dump(scaler, scaler_p)

    task.upload_artifact("processed_train", train_p)
    task.upload_artifact("processed_test", test_p)
    task.upload_artifact("scaler", scaler_p)

    task.set_parameter("Outputs/processed_dir", os.path.abspath(out_dir))
    logger.report_text(f"Processed dir: {os.path.abspath(out_dir)}")

    task.close()


if __name__ == "__main__":
    main()