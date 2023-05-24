# WandbCallback

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/keras.py#L292-L1089)

`WandbCallback`は自動的にKerasとwandbを統合します。

```python
WandbCallback(
 monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
 log_weights=(False), log_gradients=(False), save_model=(True),
 training_data=None, validation_data=None, labels=[], predictions=36,
 generator=None, input_type=None, output_type=None, log_evaluation=(False),
 validation_steps=None, class_colors=None, log_batch_frequency=None,
 log_best_prefix="best_", save_graph=(True), validation_indexes=None,
 validation_row_processor=None, prediction_row_processor=None,
 infer_missing_processors=(True), log_evaluation_frequency=0,
 compute_flops=(False), **kwargs
)
```

#### 例：

```python
model.fit(
 X_train,
 y_train,
 validation_data=(X_test, y_test),
 callbacks=[WandbCallback()],
)
```
## メソッド

### `get_flops`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/keras.py#L1043-L1089)

```python
get_flops() -> float
```

推論モードでのtf.keras.Modelまたはtf.keras.SequentialモデルのFLOPS[GFLOPs]を計算します。内部ではtf.compat.v1.profilerを使用しています。

### `set_model`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/keras.py#L554-L563)

```python
set_model(
 model
)
```
### `set_params`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/keras.py#L551-L552)

```python
set_params(
  params
)
```