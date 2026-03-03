import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    raw_dataset_id = os.environ["RAW_DATASET_ID"]

    task = Task.init(project_name=project, task_name="TEMPLATE - preprocess")
    logger = task.get_logger()

    # 1️⃣ Pull raw dataset
    raw_root = Dataset.get(dataset_id=raw_dataset_id).get_local_copy()

    df = pd.read_csv(os.path.join(raw_root, "iris.csv"))

    y = df["class"].values
    X = df.drop(columns=["class"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"X": X_train, "y": y_train}, "data/processed/train.joblib")
    joblib.dump({"X": X_test, "y": y_test}, "data/processed/test.joblib")

    # 2️⃣ Create NEW processed dataset version
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    processed_ds = Dataset.create(
        dataset_project=project,
        dataset_name="iris_processed",
        dataset_version=version,
    )

    processed_ds.add_files("data/processed")
    processed_ds.upload()
    processed_ds.finalize()

    logger.report_text(f"Created processed dataset: {processed_ds.id}")

    task.set_parameter("Outputs/processed_dataset_id", processed_ds.id)
    task.close()


if __name__ == "__main__":
    main()