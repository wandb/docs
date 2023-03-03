# WandbMetricsLogger



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/integration/keras/callbacks/metrics_logger.py#L23-L129)



`WandbMetricsLogger` automatically logs the `logs` dictionary

```python
WandbMetricsLogger(
 log_freq: Union[LogStrategy, int] = "epoch",
 initial_global_step: int = 0,
 \*args,
 **kwargs
) -> None
```



that callback methods take as argument to wandb.

This callback automatically logs the following to a W&B run page:
\* system (CPU/GPU/TPU) metrics,
\* train and validation metrics defined in `model.compile`,
\* learning rate (both for a fixed value or a learning rate scheduler)

#### Notes:


If you resume training by passing `initial_epoch` to `model.fit` and
you are using a learning rate scheduler, make sure to pass
`initial_global_step` to `WandbMetricsLogger`. The `initial_global_step`
is `step_size \* initial_step`, where `step_size` is number of training
steps per epoch. `step_size` can be calculated as the product of the
cardinality of the training dataset and the batch size.

| Arguments | |
| :--- | :--- |
| log_freq ("epoch", "batch", or int): if "epoch", logs metrics at the end of each epoch. If "batch", logs metrics at the end of each batch. If an integer, logs metrics at the end of that many batches. Defaults to "epoch". initial_global_step (int): Use this argument to correcly log the learning rate when you resume training from some `initial_epoch`, and a learning rate scheduler is used. This can be computed as `step_size \* initial_step`. Defaults to 0. |



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






