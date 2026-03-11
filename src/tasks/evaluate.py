import os
import joblib
import matplotlib.pyplot as plt
from typing import Optional

from clearml import Task, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _resolve_processed_dataset_id(project: str, processed_dataset_id: Optional[str]) -> str:
    if processed_dataset_id:
        return processed_dataset_id

    datasets = Dataset.list_datasets(
        dataset_project=project,
        partial_name="iris_preprocessed",
        tags=["preprocessed"],
        only_completed=True,
    )
    if not datasets:
        raise ValueError(
            "PROCESSED_DATASET_ID is not set and no completed preprocessed dataset was found. "
            "Run preprocess.py first or set PROCESSED_DATASET_ID."
        )

    latest = max(
        datasets,
        key=lambda ds: str(ds.get("created", "")) if isinstance(ds, dict) else str(getattr(ds, "created", "")),
    )
    dataset_id = latest.get("id") if isinstance(latest, dict) else getattr(latest, "id", None)
    if not dataset_id:
        raise ValueError("Unable to resolve processed dataset id from Dataset.list_datasets().")
    return dataset_id


def _load_test_pack(project: str, processed_dir: Optional[str], processed_dataset_id: Optional[str]) -> dict:
    if processed_dir:
        return joblib.load(os.path.join(processed_dir, "test.joblib"))

    dataset_id = _resolve_processed_dataset_id(project, processed_dataset_id)
    processed_root = Dataset.get(dataset_id=dataset_id).get_local_copy()
    return joblib.load(os.path.join(processed_root, "test.joblib"))


def _unpack_test_pack(test_pack: dict):
    if "X" in test_pack and "y" in test_pack:
        return test_pack["X"], test_pack["y"]
    if "X_test" in test_pack and "y_test" in test_pack:
        return test_pack["X_test"], test_pack["y_test"]
    raise KeyError("test.joblib does not contain expected keys (X/y or X_test/y_test).")


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    processed_dir = os.environ.get("PROCESSED_DIR")
    processed_dataset_id = os.environ.get("PROCESSED_DATASET_ID")
    model_path = os.environ.get("MODEL_PATH", "artifacts/iris_logreg.joblib")

    task = Task.init(project_name=project, task_name="TEMPLATE - evaluate", reuse_last_task_id=False)
    logger = task.get_logger()

    test_pack = _load_test_pack(project, processed_dir, processed_dataset_id)
    X_test, y_test = _unpack_test_pack(test_pack)

    model = joblib.load(model_path)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, preds))
    logger.report_scalar("accuracy", "test", value=acc, iteration=0)

    try:
        if probs.shape[1] == 2:
            auc = float(roc_auc_score(y_test, probs[:, 1]))
        else:
            auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
        logger.report_scalar("roc_auc", "test", value=auc, iteration=0)
    except ValueError as ex:
        logger.report_text(f"roc_auc not reported: {ex}", print_console=False)

    classes = getattr(model, "classes_", None)
    if classes is None:
        cm = confusion_matrix(y_test, preds)
        labels = sorted({str(v) for v in y_test})
    else:
        cm = confusion_matrix(y_test, preds, labels=classes)
        labels = [str(c) for c in classes]

    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="test",
        matrix=cm,
        iteration=0,
        xaxis="Predicted",
        yaxis="True",
        xlabels=labels,
        ylabels=labels,
        yaxis_reversed=True,
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")

    out_dir = ensure_dir("reports")
    plot_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.report_image("confusion_matrix", "test", iteration=0, local_path=plot_path)
    task.upload_artifact("confusion_matrix_png", plot_path)
    task.flush(wait_for_uploads=True)
    task.close()


if __name__ == "__main__":
    main()
