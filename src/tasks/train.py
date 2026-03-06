import os
import joblib
from typing import Optional

from clearml import Task, Dataset, OutputModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


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
    task.connect({"C": C, "max_iter": max_iter}, name="hparams")

    # 1) Pull processed dataset locally
    processed_root = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()

    # 2) Load train/test files
    train_pack = joblib.load(os.path.join(processed_root, "train.joblib"))
    test_pack = joblib.load(os.path.join(processed_root, "test.joblib"))

    X_train, y_train = train_pack["X"], train_pack["y"]
    X_test, y_test = test_pack["X"], test_pack["y"]

    # 3) Train model
    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, preds))
    ll = float(log_loss(y_test, probs))

    logger.report_scalar("accuracy", "test", value=acc, iteration=0)
    logger.report_scalar("log_loss", "test", value=ll, iteration=0)

    # 4) Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.abspath("artifacts/iris_logreg.joblib")
    joblib.dump(model, model_path)

    task.upload_artifact("trained_model", model_path)

    om = OutputModel(
        task=task,
        name="iris_logreg",
        framework="scikit-learn",
        tags=["demo"],
    )
    om.update_weights(model_path)
    om.publish()

    task.set_parameter("Outputs/model_path", model_path)
    task.close()


if __name__ == "__main__":
    main()
