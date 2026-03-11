import argparse
import os

from clearml.automation.controller import PipelineController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default=os.environ.get("CLEARML_PROJECT", "ClearML_GHA_Demo"))
    parser.add_argument("--name", type=str, default=os.environ.get("CLEARML_PIPELINE_NAME", "Iris_Pipeline"))
    parser.add_argument("--queue", type=str, default=os.environ.get("CLEARML_QUEUE", "default"))
    args = parser.parse_args()

    pipe = PipelineController(
        name=args.name,
        project=args.project,
        version="1.0.0",
        add_pipeline_tags=True,
    )

    # These TEMPLATE tasks must exist in ClearML already (run each once to register)
    base_project = args.project

    pipe.add_step(
        name="ingest",
        base_task_project=base_project,
        base_task_name="TEMPLATE - ingest",
        execution_queue=args.queue,
    )

    pipe.add_step(
        name="preprocess",
        base_task_project=base_project,
        base_task_name="TEMPLATE - preprocess",
        execution_queue=args.queue,
        parents=["ingest"],
        parameter_override={
            "Environment/Variables/RAW_DATASET_ID": "${ingest.Parameters.Outputs/dataset_id}",
        },
    )

    pipe.add_step(
        name="train",
        base_task_project=base_project,
        base_task_name="TEMPLATE - train",
        execution_queue=args.queue,
        parents=["preprocess"],
        parameter_override={
            "Environment/Variables/PROCESSED_DATASET_ID": "${preprocess.Parameters.Outputs/processed_dataset_id}",
            "Environment/Variables/C": "1.0",
            "Environment/Variables/MAX_ITER": "200",
            "Environment/Variables/LOG_EVERY": "10",
        },
    )

    pipe.add_step(
        name="evaluate",
        base_task_project=base_project,
        base_task_name="TEMPLATE - evaluate",
        execution_queue=args.queue,
        parents=["train", "preprocess"],
        parameter_override={
            "Environment/Variables/PROCESSED_DATASET_ID": "${preprocess.Parameters.Outputs/processed_dataset_id}",
            "Environment/Variables/MODEL_PATH": "${train.Parameters.Outputs/model_path}",
            "Environment/Variables/EVAL_LOG_EVERY": "5",
        },
    )

    pipe.start(queue=args.queue)
    pipe.wait()
    pipe.stop()


if __name__ == "__main__":
    main()
