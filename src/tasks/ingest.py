# src/tasks/ingest.py
import os
from datetime import datetime, timezone

import pandas as pd
from clearml import Task, Dataset
from sklearn.datasets import fetch_openml


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    name = os.environ.get("DATASET_NAME", "iris_openml")

    task = Task.init(project_name=project, task_name="TEMPLATE - ingest")
    task.set_packages(requirements_file="requirements.txt")
    log = task.get_logger()

    # 1) Download dataset (from the web)
    os.makedirs("data/raw", exist_ok=True)
    csv_path = os.path.join("data/raw", "iris.csv")
    df = fetch_openml(name="iris", version=1, as_frame=True).frame
    df.to_csv(csv_path, index=False)
    log.report_text(f"Saved {csv_path}")

    # 2) Version it as a ClearML Dataset
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ds = Dataset.create(dataset_project=project, dataset_name=name, dataset_version=version)
    ds.add_files("data/raw")
    ds.upload()
    ds.finalize()

    # 3) Output dataset id for the pipeline
    task.set_parameter("Outputs/dataset_id", ds.id)
    log.report_text(f"Dataset id: {ds.id}")

    task.close()


if __name__ == "__main__":
    main()