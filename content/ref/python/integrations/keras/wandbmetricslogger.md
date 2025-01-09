---
title: WandbMetricsLogger
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/integration/keras/callbacks/metrics_logger.py#L23-L136" >}}

Logger that sends system metrics to W&B.

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger` automatically logs the `logs` dictionary that callback methods
take as argument to wandb.

This callback automatically logs the following to a W&B run page:

* system (CPU/GPU/TPU) metrics,
* train and validation metrics defined in `model.compile`,
* learning rate (both for a fixed value or a learning rate scheduler)

#### Notes:

If you resume training by passing `initial_epoch` to `model.fit` and you are using a
learning rate scheduler, make sure to pass `initial_global_step` to
`WandbMetricsLogger`. The `initial_global_step` is `step_size * initial_step`, where
`step_size` is number of training steps per epoch. `step_size` can be calculated as
the product of the cardinality of the training dataset and the batch size.

| Args |  |
| :--- | :--- |
|  `log_freq` |  (`epoch`, `batch`, or an `int`) if `epoch`, logs metrics at the end of each epoch. If `batch`, logs metrics at the end of each batch. If an `int`, logs metrics at the end of that many batches. Defaults to `epoch`. |
|  `initial_global_step` |  (int) Use this argument to correctly log the learning rate when you resume training from some `initial_epoch`, and a learning rate scheduler is used. This can be computed as `step_size * initial_step`. Defaults to 0. |

## Methods

### `set_model`

```python
set_model(
    model
)
```

### `set_params`

```python
set_params(
    params
)
```
