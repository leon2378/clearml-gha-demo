import os
import joblib
import matplotlib.pyplot as plt

from clearml import Task
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def main():
    project = os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo")
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")
    model_path = os.environ.get("MODEL_PATH", "artifacts/logreg.joblib")

    task = Task.init(project_name=project, task_name="TEMPLATE - evaluate", reuse_last_task_id=False)
    task.set_packages(requirements_file="requirements.txt")
    logger = task.get_logger()

    test_pack = joblib.load(os.path.join(processed_dir, "test.joblib"))
    X_test, y_test = test_pack["X_test"], test_pack["y_test"]

    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))
    logger.report_scalar("accuracy", "test", 0, acc)
    logger.report_scalar("roc_auc", "test", 0, auc)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    disp.plot(ax=ax, values_format="d")

    out_dir = ensure_dir("reports")
    plot_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.report_image("confusion_matrix", "test", 0, local_path=plot_path)
    task.upload_artifact("confusion_matrix_png", plot_path)
    task.close()


if __name__ == "__main__":
    main()