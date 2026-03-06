import os
import joblib
import pandas as pd
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    raw_dataset_id = os.environ.get("RAW_DATASET_ID")

    if not raw_dataset_id:
        raise ValueError("RAW_DATASET_ID is not set")

    task = Task.init(project_name=project, task_name="TEMPLATE - preprocess")
    logger = task.get_logger()

    logger.report_text(f"Using raw dataset id: {raw_dataset_id}")

    # 1) Pull raw dataset
    raw_root = Dataset.get(dataset_id=raw_dataset_id).get_local_copy()
    csv_path = os.path.join(raw_root, "iris.csv")

    df = pd.read_csv(csv_path)

    y = df["class"].values
    X = df.drop(columns=["class"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2) Save processed files
    os.makedirs("data/processed", exist_ok=True)

    train_path = os.path.abspath("data/processed/train.joblib")
    test_path = os.path.abspath("data/processed/test.joblib")
    scaler_path = os.path.abspath("data/processed/scaler.joblib")

    joblib.dump({"X": X_train, "y": y_train}, train_path)
    joblib.dump({"X": X_test, "y": y_test}, test_path)
    joblib.dump(scaler, scaler_path)

    # 3) Create new processed dataset version
    processed_ds = Dataset.create(
        dataset_project=project,
        dataset_name="iris_preprocessed",
        parent_datasets=[raw_dataset_id],
        dataset_tags=["preprocessed"],
        dataset_version="1.2",
    )

    processed_ds.add_files("data/processed")
    processed_ds.upload(show_progress=True, verbose=True)
    processed_ds.finalize()

    logger.report_text(f"Created processed dataset: {processed_ds.id}")

    # 4) Output dataset id for next pipeline step
    task.set_parameter("Outputs/processed_dataset_id", processed_ds.id)

    task.close()


if __name__ == "__main__":
    main()