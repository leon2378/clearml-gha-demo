import os
import joblib
import numpy as np

from clearml import Task, OutputModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")

    task = Task.init(project_name=project, task_name="TEMPLATE - train", reuse_last_task_id=False)
    task.set_packages(requirements_file="requirements.txt")
    logger = task.get_logger()

    C = float(os.environ.get("C", "1.0"))
    max_iter = int(os.environ.get("MAX_ITER", "200"))
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

    logger.report_scalar("accuracy", "test", 0, acc)
    logger.report_scalar("roc_auc", "test", 0, auc)
    logger.report_scalar("log_loss", "test", 0, ll)

    out_dir = ensure_dir("artifacts")
    model_path = os.path.join(out_dir, "logreg.joblib")
    joblib.dump(model, model_path)
    task.upload_artifact("trained_model", model_path)

    om = OutputModel(task=task, name="breast_cancer_logreg", framework="scikit-learn", tags=["demo"])
    om.update_weights(model_path)
    om.publish()

    task.set_parameter("Outputs/model_path", os.path.abspath(model_path))
    task.close()


if __name__ == "__main__":
    main()