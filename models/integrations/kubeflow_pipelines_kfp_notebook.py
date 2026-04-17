# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wandb>=0.12.11",
#     "kfp>=1.8,<2.0",
#     "kubernetes",
# ]
# ///

"""Marimo notebook: code from the Kubeflow Pipelines (kfp) integration doc."""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from textwrap import dedent

    def md(body: str):
        return mo.md(dedent(body).strip())

    return (md,)


@app.cell
def _(md):
    return md(
        """
        # Kubeflow Pipelines (kfp) and W&B

        This notebook mirrors the code on the
        [Kubeflow Pipelines (kfp)](https://docs.wandb.ai/models/integrations/kubeflow-pipelines-kfp)
        integration page. The integration requires `wandb>=0.12.11` and `kfp<2.0.0`
        (see the doc page for current compatibility notes).

        Set `WANDB_API_KEY` before running cells that call W&B. For two-way linking with Kubeflow,
        set `WANDB_KUBEFLOW_URL` to your Kubeflow Pipelines base URL when you run on a cluster.
        """
    )


@app.cell
def _(md):
    return md(
        """
        ## Install the `wandb` library and log in

        Dependencies for this file are declared in the PEP 723 header. Log in from Python:
        """
    )


@app.cell
def _():
    import wandb

    wandb.login()
    return (wandb,)


@app.cell
def _(md):
    return md(
        """
        ## Decorate your components

        Add the `@wandb_log` decorator and build components as usual. Inputs and outputs are logged
        to W&B when the pipeline runs.
        """
    )


@app.cell
def _(wandb):
    from kfp import components
    from wandb.integration.kfp import wandb_log

    @wandb_log
    def add(a: float, b: float) -> float:
        return a + b

    add = components.create_component_from_func(add)
    return add, components, wandb_log


@app.cell
def _(md):
    return md(
        """
        ## Pass environment variables to containers

        Pass W&B-related environment variables into pipeline ops. For two-way linking, set
        `WANDB_KUBEFLOW_URL` to your Kubeflow Pipelines base URL (for example `https://kubeflow.example.com`).
        """
    )


@app.cell
def _(add):
    import os

    from kfp import dsl
    from kubernetes.client.models import V1EnvVar

    def add_wandb_env_variables(op):
        env = {
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_BASE_URL": os.getenv("WANDB_BASE_URL"),
        }

        for name, value in env.items():
            op = op.add_env_variable(V1EnvVar(name, value))
        return op

    @dsl.pipeline(name="example-pipeline")
    def example_pipeline(a: float, b: float):
        conf = dsl.get_pipeline_conf()
        conf.add_op_transformer(add_wandb_env_variables)
        add_task = add(a=a, b=b)
        return add_task

    return add_wandb_env_variables, dsl, example_pipeline, V1EnvVar


@app.cell
def _(md):
    return md(
        """
        ## Concept mapping from Kubeflow Pipelines to W&B

        | Kubeflow Pipelines | W&B | Location in W&B |
        | --- | --- | --- |
        | Input Scalar | [`config`](https://docs.wandb.ai/models/) | [Overview tab](https://docs.wandb.ai/models/runs/#overview-tab) |
        | Output Scalar | [`summary`](https://docs.wandb.ai/models/) | [Overview tab](https://docs.wandb.ai/models/runs/#overview-tab) |
        | Input Artifact | Input artifact | [Artifacts tab](https://docs.wandb.ai/models/runs/#artifacts-tab) |
        | Output Artifact | Output artifact | [Artifacts tab](https://docs.wandb.ai/models/runs/#artifacts-tab) |
        """
    )


@app.cell
def _(md):
    return md(
        """
        ## Fine-grain logging

        For finer control, call `wandb.log` and `wandb.log_artifact` inside the component.

        ### With explicit `wandb.log` calls

        The `@wandb_log` decorator still tracks inputs and outputs. Below, a small synthetic training
        loop shows explicit `run.log` usage. Paths follow KFP `InputPath` / `OutputPath` patterns.
        """
    )


@app.cell
def _(components, wandb, wandb_log):
    from pathlib import Path

    @wandb_log
    def train_model(
        train_dataloader_path: components.InputPath("dataloader"),
        test_dataloader_path: components.InputPath("dataloader"),
        model_path: components.OutputPath("pytorch_model"),
    ):
        train_dataloader_path = Path(train_dataloader_path)
        test_dataloader_path = Path(test_dataloader_path)
        model_path = Path(model_path)

        with wandb.init() as run:
            epochs = 2
            log_interval = 1
            for epoch in range(epochs):
                for batch_idx in range(3):
                    loss = 1.0 / (batch_idx + 1)
                    if batch_idx % log_interval == 0:
                        run.log(
                            {
                                "epoch": epoch,
                                "step": batch_idx * 2,
                                "loss": loss,
                            }
                        )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text("placeholder model weights", encoding="utf-8")
            model_artifact = wandb.Artifact("model", type="model")
            model_artifact.add_file(str(model_path))
            run.log_artifact(model_artifact)

    train_model_op = components.create_component_from_func(train_model)
    return train_model, train_model_op


@app.cell
def _(md):
    return md(
        """
        ### With implicit W&B integrations (PyTorch Lightning)

        If you use a [supported framework integration](https://docs.wandb.ai/models/integrations),
        pass the logger into your trainer. Install `pytorch-lightning` in your component image or
        local environment when you use this pattern.
        """
    )


@app.cell
def _(components, wandb_log):
    @wandb_log
    def train_model_lightning(
        train_dataloader_path: components.InputPath("dataloader"),
        test_dataloader_path: components.InputPath("dataloader"),
        model_path: components.OutputPath("pytorch_model"),
    ):
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import WandbLogger

        trainer = Trainer(logger=WandbLogger())
        # Training code would go here (datasets, model, trainer.fit, and so on).

    train_model_lightning_op = components.create_component_from_func(train_model_lightning)
    return train_model_lightning, train_model_lightning_op


if __name__ == "__main__":
    app.run()
