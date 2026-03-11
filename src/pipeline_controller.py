import argparse
import os
from typing import Optional

from clearml import Task
from clearml.automation.controller import PipelineController


def _resolve_base_task_id(task_name: str, base_project: Optional[str]) -> str:
    query_name = "^{}$".format(task_name)

    tasks = []
    if base_project:
        tasks = Task.get_tasks(
            project_name=base_project,
            task_name=query_name,
            allow_archived=False,
            task_filter={"order_by": ["-last_update"]},
        )

    if not tasks:
        tasks = Task.get_tasks(
            task_name=query_name,
            allow_archived=False,
            task_filter={"order_by": ["-last_update"]},
        )

    if not tasks:
        raise ValueError(
            "Could not find base task '{}'. Run that template task once first, "
            "or pass --base-project with the correct project.".format(task_name)
        )

    return tasks[0].id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo"))
    parser.add_argument(
        "--base-project",
        type=str,
        default=os.environ.get("CLEARML_TEMPLATE_PROJECT"),
        help="Project where TEMPLATE tasks are registered. Falls back to searching all projects if not found.",
    )
    parser.add_argument("--name", type=str, default=os.environ.get("CLEARML_PIPELINE_NAME", "Iris_Pipeline"))
    parser.add_argument("--queue", type=str, default=os.environ.get("CLEARML_QUEUE", "default"))
    args = parser.parse_args()

    pipe = PipelineController(
        name=args.name,
        project=args.project,
        version="1.0.0",
        add_pipeline_tags=True,
    )

    # Resolve template task IDs once; this is more robust than relying on exact project lookup.
    ingest_base_id = _resolve_base_task_id("TEMPLATE - ingest", args.base_project)
    preprocess_base_id = _resolve_base_task_id("TEMPLATE - preprocess", args.base_project)
    train_base_id = _resolve_base_task_id("TEMPLATE - train", args.base_project)
    evaluate_base_id = _resolve_base_task_id("TEMPLATE - evaluate", args.base_project)

    pipe.add_step(
        name="ingest",
        base_task_id=ingest_base_id,
        execution_queue=args.queue,
    )

    pipe.add_step(
        name="preprocess",
        base_task_id=preprocess_base_id,
        execution_queue=args.queue,
        parents=["ingest"],
        parameter_override={
            "Environment/Variables/RAW_DATASET_ID": "${ingest.parameters.Outputs/dataset_id}",
        },
    )

    pipe.add_step(
        name="train",
        base_task_id=train_base_id,
        execution_queue=args.queue,
        parents=["preprocess"],
        parameter_override={
            "Environment/Variables/PROCESSED_DATASET_ID": "${preprocess.parameters.Outputs/processed_dataset_id}",
            "Environment/Variables/C": "1.0",
            "Environment/Variables/MAX_ITER": "200",
            "Environment/Variables/LOG_EVERY": "10",
        },
    )

    pipe.add_step(
        name="evaluate",
        base_task_id=evaluate_base_id,
        execution_queue=args.queue,
        parents=["train", "preprocess"],
        parameter_override={
            "Environment/Variables/PROCESSED_DATASET_ID": "${preprocess.parameters.Outputs/processed_dataset_id}",
            "Environment/Variables/MODEL_PATH": "${train.parameters.Outputs/model_path}",
            "Environment/Variables/EVAL_LOG_EVERY": "5",
        },
    )

    pipe.start(queue=args.queue)
    pipe.wait()
    pipe.stop()


if __name__ == "__main__":
    main()
