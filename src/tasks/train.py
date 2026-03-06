import os
import joblib
import warnings
from typing import Optional

from clearml import Task, Dataset, OutputModel
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    processed_dataset_id = _resolve_processed_dataset_id(
        project=project,
        processed_dataset_id=os.environ.get("PROCESSED_DATASET_ID"),
    )

    task = Task.init(project_name=project, task_name="TEMPLATE - train", reuse_last_task_id=False)
    logger = task.get_logger()

    logger.report_text(f"Using processed dataset id: {processed_dataset_id}")

    C = float(os.environ.get("C", "1.0"))
    max_iter = int(os.environ.get("MAX_ITER", "200"))
    log_every = max(1, int(os.environ.get("LOG_EVERY", "10")))
    task.connect({"C": C, "max_iter": max_iter, "log_every": log_every}, name="hparams")

    # 1) Pull processed dataset locally
    processed_root = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()

    # 2) Load train/test files
    train_pack = joblib.load(os.path.join(processed_root, "train.joblib"))
    test_pack = joblib.load(os.path.join(processed_root, "test.joblib"))

    X_train, y_train = train_pack["X"], train_pack["y"]
    X_test, y_test = test_pack["X"], test_pack["y"]

    # 3) Train model
    model = LogisticRegression(C=C, max_iter=1, solver="lbfgs", warm_start=True)
    preds = None
    probs = None

    for step in range(1, max_iter + 1):
        # We intentionally run one solver step per fit() to report a scalar curve over iterations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)

        if step == 1 or step % log_every == 0 or step == max_iter:
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            acc = float(accuracy_score(y_test, preds))
            ll = float(log_loss(y_test, probs))
            logger.report_scalar("accuracy", "test", value=acc, iteration=step)
            logger.report_scalar("log_loss", "test", value=ll, iteration=step)

    if preds is None or probs is None:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

    labels = [str(c) for c in model.classes_]
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="train",
        matrix=cm,
        iteration=0,
        xaxis="Predicted",
        yaxis="True",
        xlabels=labels,
        ylabels=labels,
        yaxis_reversed=True,
    )

    # 4) Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.abspath("artifacts/iris_logreg.joblib")
    joblib.dump(model, model_path)

    # Debug Samples tab: upload rendered confusion matrix image
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=True)
    fig.tight_layout()
    cm_path = os.path.abspath("artifacts/confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.report_image("confusion_matrix", "train", iteration=0, local_path=cm_path)

    task.upload_artifact("trained_model", model_path)
    task.upload_artifact("confusion_matrix_png", cm_path)

    om = OutputModel(
        task=task,
        name="iris_logreg",
        framework="scikit-learn",
        tags=["demo"],
    )
    om.update_weights(model_path)
    om.publish()

    task.set_parameter("Outputs/model_path", model_path)
    task.flush(wait_for_uploads=True)
    task.close()


if __name__ == "__main__":
    main()
