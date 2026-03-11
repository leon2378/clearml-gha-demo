import os
import joblib
import matplotlib.pyplot as plt
from typing import Optional

from clearml import Task, Dataset
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay


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


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    processed_dir = os.environ.get("PROCESSED_DIR")
    model_path = os.environ.get("MODEL_PATH", "artifacts/iris_logreg.joblib")
    eval_log_every = max(1, int(os.environ.get("EVAL_LOG_EVERY", "5")))

    if processed_dir:
        processed_root = processed_dir
    else:
        processed_dataset_id = _resolve_processed_dataset_id(
            project=project,
            processed_dataset_id=os.environ.get("PROCESSED_DATASET_ID"),
        )
        processed_root = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()

    task = Task.init(
        project_name=project,
        task_name="TEMPLATE - evaluate",
        reuse_last_task_id=False,
        auto_connect_frameworks={"matplotlib": False},
    )
    logger = task.get_logger()
    task.connect({"eval_log_every": eval_log_every}, name="hparams")

    test_pack = joblib.load(os.path.join(processed_root, "test.joblib"))
    X_test, y_test = test_pack["X"], test_pack["y"]

    model = joblib.load(model_path)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Log running metrics so the scalar chart has multiple points.
    n = len(y_test)
    #n = 100
    for i in range(eval_log_every, n + 1, eval_log_every):
        acc_i = float(accuracy_score(y_test[:i], preds[:i]))
        ll_i = float(log_loss(y_test[:i], probs[:i], labels=model.classes_))
        logger.report_scalar("accuracy", "test", value=acc_i, iteration=i)
        logger.report_scalar("log_loss", "test", value=ll_i, iteration=i)
    if n % eval_log_every != 0:
        acc_i = float(accuracy_score(y_test, preds))
        ll_i = float(log_loss(y_test, probs, labels=model.classes_))
        logger.report_scalar("accuracy", "test", value=acc_i, iteration=n)
        logger.report_scalar("log_loss", "test", value=ll_i, iteration=n)

    labels = [str(c) for c in model.classes_]
    cm = confusion_matrix(y_test, preds, labels=model.classes_)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=True)
    fig.tight_layout()
    logger.report_matplotlib_figure(
        title="confusion_matrix",
        series="test",
        figure=fig,
        iteration=n,
        report_image=False,
    )

    os.makedirs("reports", exist_ok=True)
    cm_path = os.path.abspath("reports/test_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    task.upload_artifact("test_confusion_matrix", cm_path)
    #task.upload_artifact("best_model", model_path)
    task.flush(wait_for_uploads=True)
    task.close()


if __name__ == "__main__":
    main()
