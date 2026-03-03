import os
from datetime import datetime, timezone

from clearml import Task, Dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    dataset_name = os.environ.get("DATASET_NAME", "breast_cancer_small")

    task = Task.init(project_name=project, task_name="TEMPLATE - ingest", reuse_last_task_id=False)
    task.set_packages(requirements_file="requirements.txt")
    logger = task.get_logger()

    out_dir = ensure_dir("data/raw")
    df = load_breast_cancer(as_frame=True).frame
    csv_path = os.path.join(out_dir, "breast_cancer.csv")
    df.to_csv(csv_path, index=False)

    logger.report_text(f"Saved raw CSV to: {csv_path}")

    # Try ClearML Dataset, fallback to artifact
    try:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ds = Dataset.create(
            dataset_project=project,
            dataset_name=dataset_name,
            dataset_version=version,
        )
        ds.add_files(out_dir)
        ds.upload(show_progress=True)
        ds.finalize()
        task.set_parameter("Outputs/dataset_id", ds.id)
        task.set_parameter("Outputs/local_path", "")
        logger.report_text(f"Created ClearML Dataset: {ds.id}")
    except Exception as e:
        logger.report_text(f"Dataset create/upload failed, falling back to artifact. Error: {repr(e)}")
        task.upload_artifact("raw_data_csv", csv_path)
        task.set_parameter("Outputs/dataset_id", "")
        task.set_parameter("Outputs/local_path", os.path.abspath(out_dir))

    task.close()


if __name__ == "__main__":
    main()