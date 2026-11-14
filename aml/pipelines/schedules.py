"""ParkWatch — Azure ML schedules that close the loop, Madrid time.

  parkwatch-training-nightly  02:00 every day except Monday   training (epochs of the last run with CV)
  parkwatch-training-weekly   02:00 on Mondays (Sunday night) full cross-validation, forced
  parkwatch-inference         04:00 every day                 champion -> Label Studio, creates new tasks

Each schedule stores a copy of the code, the components and rois.json as they were when it was
created (the landmark anchors are read from Label Studio every night): after changing any of them, run this script again to update them.

Usage (from the repo root, with az login):
  uv run python aml/pipelines/schedules.py            # creates or updates the three schedules
  uv run python aml/pipelines/schedules.py --disable  # pauses them (nothing is deleted)
"""
import argparse

import training  # noqa: F401  (enables the SDK preview features before azure.ai.ml is imported)
from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import TimeZone
from azure.ai.ml.entities import CronTrigger, JobSchedule
from azure.identity import DefaultAzureCredential

import inference

MADRID = TimeZone.ROMANCE_STANDARD_TIME


def schedules(client: MLClient) -> list[JobSchedule]:
    client_id = training.cluster_client_id(client, "cpu-cluster")
    rois = Input(type="uri_file", path=str(training.ROIS))

    def training_job(force: bool, run_cv: bool):
        job = training.build_pipeline(client_id)(rois=rois, force=force, run_cv=run_cv)
        return training.configure(job, "parkwatch-training")

    infer = training.configure(
        inference.build_pipeline(client_id, days=[])(rois=rois, create_missing=True),
        "parkwatch-inference",
    )

    return [
        JobSchedule(
            name="parkwatch-training-nightly",
            description="Retrains only if labels, ROIs or alignment changed; epochs of the last run with CV.",
            trigger=CronTrigger(expression="0 2 * * 0,2-6", time_zone=MADRID),
            create_job=training_job(force=False, run_cv=False),
        ),
        JobSchedule(
            name="parkwatch-training-weekly",
            description="Weekly full cross-validation, even without new labels.",
            trigger=CronTrigger(expression="0 2 * * 1", time_zone=MADRID),
            create_job=training_job(force=True, run_cv=True),
        ),
        JobSchedule(
            name="parkwatch-inference",
            description="The champion predicts the last 3 days and creates/updates the Label Studio tasks.",
            trigger=CronTrigger(expression="0 4 * * *", time_zone=MADRID),
            create_job=infer,
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--disable", action="store_true", help="pause the schedules")
    args = p.parse_args()

    client = MLClient(DefaultAzureCredential(), training.SUBSCRIPTION, training.RESOURCE_GROUP, training.WORKSPACE)
    for schedule in schedules(client):
        if args.disable:
            client.schedules.begin_disable(schedule.name).result()
            print(f"paused: {schedule.name}")
            continue
        created = client.schedules.begin_create_or_update(schedule).result()
        print(f"{created.name}: {created.trigger.expression} ({created.trigger.time_zone}) enabled={created.is_enabled}")


if __name__ == "__main__":
    main()
