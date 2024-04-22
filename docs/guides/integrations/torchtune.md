---
displayed_sidebar: default
---


# torchtune

[**Check our blog post →**](https://wandb.ai/capecape/torchtune-mistral/reports/torchtune-The-new-PyTorch-LLM-fine-tuning-library---Vmlldzo3NTUwNjM0)

[torchtune](https://pytorch.org/torchtune/stable/index.html) is a PyTorch-based library designed to streamline the authoring, fine-tuning, and experimentation processes for large language models (LLMs). It focuses on simplicity and extensibility with a componentized, native-PyTorch design that avoids unnecessary abstractions and promotes easy reuse. The library ensures correctness and stability, setting high standards for component testing and performance benchmarks. Torchtune is built to democratize LLM fine-tuning, offering out-of-the-box functionality across different hardware setups. It features modular implementations of popular LLMs, interoperability through checkpoint conversion, and a variety of training recipes. The library also integrates with external datasets and evaluation tools, supports distributed training, and allows configuration through YAML files to adjust training settings and hyperparameters without modifying code. Additionally, torchtune has built-in support for [logging with Weights & Biases](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html), enhancing tracking and visualization of training processes. With a focus on usability, torchtune adheres to PyTorch’s philosophy by emphasizing clarity and modular building blocks over monolithic design, aiming to make fine-tuning accessible and efficient for developers.

:::info
Check the [torchtune documentation](https://pytorch.org/torchtune/stable/deep_dives/wandb_logging.html) on how to use Weights & Biases to tune your LLM models.
:::


## Using the Weights & Biases metric logger

You can quickly try the W&B integration with torchtune by overriding arguments at launch time.

1. First install the `wandb` library:

```bash
pip install -U wandb
```

2. Then, run any default recipe and pass the `metric_logger._component_` with the corresponding `WandBLogger` class. You can also pass a `project` name and `log_every_n_steps` to log metrics every n steps.

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  log_every_n_steps=5
```

The `WandBLogger` class is a subclass of `MetricLogger` and inherits all of its methods. It adds the ability to log metrics to Weights & Biases. It also supports any other `kwargs` to pass to the `wandb.init` method. For example, you can pass:

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device \
  metric_logger._component_=torchtune.utils.metric_logging.WandBLogger \
  metric_logger.project="llama3_lora" \
  metric_logger.entity="my_project" \
  metric_logger.job_type="lora_finetune_single_device" \
  metric_logger.group="my_awesome_experiments" \
  log_every_n_steps=5
```

## What do we log?

After running the above command, you can explore the W&B dashboard to see the logged metrics. By default we grab all the hyperparameters from the config file and the launch override ones.

We capture the resolved config for you on the Overview tab. We also store the config as a YAML on the [Files tab](https://wandb.ai/capecape/torchtune/runs/joyknwwa/files).

The actual computation of training metrics is inside the [recipe file](https://github.com/pytorch/torchtune/tree/main/recipes) on the [`train`](https://github.com/pytorch/torchtune/blob/cd779783f9acecccbebc3c50265f6caf97fa99aa/recipes/full_finetune_single_device.py#L374) function. Having all the metric logic on a single place makes it easier for the user to add custom metrics or modify the existing ones.

:::info
This is a fast evolving library, the current metrics are subject to change. If you want to add a custom metric, you should modify the recipe and call the corresponding `self._metric_logger.*` function.
:::

## Saving and loading checkpoints

The torchtune library supports various [checkpoint formats](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html). Depending on the origin of the model you are using, you should switch to the appropriate [checkpointer class](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html).

If you want to save the model checkpoints to [W&B Artifacts](https://docs.wandb.ai/guides/artifacts), the simples solution is to override the [`save_checkpoint`](https://github.com/pytorch/torchtune/blob/cd779783f9acecccbebc3c50265f6caf97fa99aa/recipes/full_finetune_single_device.py#L348) functions inside the corresponding recipe. 

Here is an example of how you can override the `save_checkpoint` function to save the model checkpoints to W&B Artifacts.

```python
def save_checkpoint(self, epoch: int) -> None:
    ...
    ## Let's save the checkpoint to W&B
    ## depending on the Checkpointer Class the file will be named differently
    ## Here is an example for the full_finetune case
    checkpoint_file = Path.joinpath(
        self._checkpointer._output_dir, f"torchtune_model_{epoch}"
    ).with_suffix(".pt")
    wandb_at = wandb.Artifact(
        name=f"torchtune_model_{epoch}",
        type="model",
        # description of the model checkpoint
        description="Model checkpoint",
        # you can add whatever metadata you want as a dict
        metadata={
            utils.SEED_KEY: self.seed,
            utils.EPOCHS_KEY: self.epochs_run,
            utils.TOTAL_EPOCHS_KEY: self.total_epochs,
            utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
        }
    )
    wandb_at.add_file(checkpoint_file)
    wandb.log_artifact(wandb_at)
```

:::info
We are working on adding support for loading checkpoints from W&B Artifacts.
:::


