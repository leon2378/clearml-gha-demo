import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from clearml import PipelineDecorator


@dataclass
class DataRef:
    """
    Represents either a ClearML Dataset (dataset_id) or a local path fallback.
    """
    dataset_id: Optional[str] = None
    local_path: Optional[str] = None


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@PipelineDecorator.component(cache=False, execution_queue="default")
def ingest_data(project: str, dataset_name: str = "breast_cancer_small") -> Dict[str, Any]:
    """
    Creates a tiny tabular dataset from sklearn and (if supported) versions it via ClearML Datasets.
    Fallback: returns local path when server does not support Datasets.
    """
    from clearml import Task, Dataset
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    from datetime import datetime

    task = Task.init(project_name=project, task_name="ingest_data", reuse_last_task_id=False)
    logger = task.get_logger()

    out_dir = _ensure_dir("data/raw")
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # includes target
    csv_path = os.path.join(out_dir, "breast_cancer.csv")
    df.to_csv(csv_path, index=False)

    logger.report_text(f"Saved raw CSV to: {csv_path}")
    logger.report_scalar("rows", "raw", iteration=0, value=len(df))
    logger.report_scalar("cols", "raw", iteration=0, value=df.shape[1])

    # Try to create a ClearML Dataset (may fail on older servers)
    try:
        version = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        ds = Dataset.create(
            dataset_project=project,
            dataset_name=dataset_name,
            dataset_version=version,
        )
        ds.add_files(out_dir)
        ds.upload(show_progress=True, verbose=True)
        ds.finalize()

        logger.report_text(f"Created ClearML Dataset: {ds.id}")
        return {"dataset_id": ds.id, "local_path": None}

    except NotImplementedError as e:
        # Older server (no datasets)
        logger.report_text(
            "ClearML Datasets not supported on this server. Falling back to local path.\n"
            f"Error: {repr(e)}"
        )
        # Still upload the raw CSV as an artifact for visibility
        task.upload_artifact(name="raw_data_csv", artifact_object=csv_path)
        return {"dataset_id": None, "local_path": os.path.abspath(out_dir)}


@PipelineDecorator.component(cache=False, execution_queue="default")
def preprocess_data(project: str, data_ref: Dict[str, Any], test_size: float = 0.2, random_state: int = 42) -> str:
    """
    Loads raw CSV (from ClearML Dataset or local path), splits train/test, scales features,
    saves processed artifacts.
    Returns the processed directory path (local).
    """
    from clearml import Task, Dataset
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    task = Task.init(project_name=project, task_name="preprocess_data", reuse_last_task_id=False)
    logger = task.get_logger()

    dataset_id = data_ref.get("dataset_id")
    local_path = data_ref.get("local_path")

    if dataset_id:
        ds = Dataset.get(dataset_id=dataset_id)
        raw_root = ds.get_local_copy()
    else:
        raw_root = local_path

    csv_path = os.path.join(raw_root, "breast_cancer.csv")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    out_dir = _ensure_dir("data/processed")
    joblib.dump({"X_train": X_train_scaled, "y_train": y_train}, os.path.join(out_dir, "train.joblib"))
    joblib.dump({"X_test": X_test_scaled, "y_test": y_test}, os.path.join(out_dir, "test.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    logger.report_scalar("train_size", "split", iteration=0, value=len(y_train))
    logger.report_scalar("test_size", "split", iteration=0, value=len(y_test))

    task.upload_artifact("processed_train", os.path.join(out_dir, "train.joblib"))
    task.upload_artifact("processed_test", os.path.join(out_dir, "test.joblib"))
    task.upload_artifact("scaler", os.path.join(out_dir, "scaler.joblib"))

    return os.path.abspath(out_dir)


@PipelineDecorator.component(cache=False, execution_queue="default")
def train_model(project: str, processed_dir: str, C: float = 1.0, max_iter: int = 200) -> Dict[str, Any]:
    """
    Trains a Logistic Regression model, logs metrics, uploads model artifact,
    and publishes to ClearML Model Registry.
    Returns: dict with model_path + training task id.
    """
    from clearml import Task, OutputModel
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    task = Task.init(project_name=project, task_name="train_model", reuse_last_task_id=False)
    logger = task.get_logger()

    # Log hyperparams
    task.connect({"C": C, "max_iter": max_iter}, name="hparams")

    train_pack = joblib.load(os.path.join(processed_dir, "train.joblib"))
    test_pack = joblib.load(os.path.join(processed_dir, "test.joblib"))

    X_train, y_train = train_pack["X_train"], train_pack["y_train"]
    X_test, y_test = test_pack["X_test"], test_pack["y_test"]

    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))
    ll = float(log_loss(y_test, np.vstack([1 - probs, probs]).T))

    logger.report_scalar("accuracy", "test", iteration=0, value=acc)
    logger.report_scalar("roc_auc", "test", iteration=0, value=auc)
    logger.report_scalar("log_loss", "test", iteration=0, value=ll)

    out_dir = _ensure_dir("artifacts")
    model_path = os.path.join(out_dir, "logreg.joblib")
    joblib.dump(model, model_path)

    # Upload as artifact
    task.upload_artifact("trained_model", model_path)

    # Publish to Model Registry
    output_model = OutputModel(
        task=task,
        name="breast_cancer_logreg",
        framework="scikit-learn",
        tags=["demo", "github-actions"],
    )
    output_model.update_weights(model_path)
    output_model.publish()

    logger.report_text("Published model to ClearML Model Registry.")

    return {"model_path": os.path.abspath(model_path), "train_task_id": task.id}


@PipelineDecorator.component(cache=False, execution_queue="default")
def evaluate_model(project: str, processed_dir: str, model_path: str) -> str:
    """
    Evaluates trained model and logs confusion matrix plot.
    Returns the path to the plot.
    """
    from clearml import Task
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score

    task = Task.init(project_name=project, task_name="evaluate_model", reuse_last_task_id=False)
    logger = task.get_logger()

    test_pack = joblib.load(os.path.join(processed_dir, "test.joblib"))
    X_test, y_test = test_pack["X_test"], test_pack["y_test"]

    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    logger.report_scalar("accuracy", "test", iteration=0, value=acc)
    logger.report_scalar("roc_auc", "test", iteration=0, value=auc)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")

    out_dir = _ensure_dir("reports")
    plot_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Report image to ClearML + upload artifact
    logger.report_image("confusion_matrix", "test", iteration=0, local_path=plot_path)
    task.upload_artifact("confusion_matrix_png", plot_path)

    return os.path.abspath(plot_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo"))
    parser.add_argument("--name", type=str, default=os.environ.get("CLEARML_PIPELINE_NAME", "BreastCancer_Pipeline"))
    args = parser.parse_args()

    @PipelineDecorator.pipeline(
        name=args.name,
        project=args.project,
        version="1.0.0",
        pipeline_execution_queue="default",
    )
    def pipeline():
        ref = ingest_data(project=args.project)
        processed = preprocess_data(project=args.project, data_ref=ref)
        train_out = train_model(project=args.project, processed_dir=processed, C=1.0, max_iter=200)
        evaluate_model(project=args.project, processed_dir=processed, model_path=train_out["model_path"])

    pipeline()


if __name__ == "__main__":
    main()