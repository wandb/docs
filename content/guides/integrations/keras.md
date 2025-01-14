---
menu:
  default:
    identifier: keras
    parent: integrations
title: Keras
weight: 160
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb" >}}

## Keras callbacks

W&B has three callbacks for Keras, available from `wandb` v0.13.4. For the legacy `WandbCallback` scroll down.


- **`WandbMetricsLogger`** : Use this callback for [Experiment Tracking](/guides/track). It logs your training and validation metrics along with system metrics to Weights and Biases.

- **`WandbModelCheckpoint`** : Use this callback to log your model checkpoints to Weight and Biases [Artifacts](/guides/artifacts).

- **`WandbEvalCallback`**: This base callback logs model predictions to Weights and Biases [Tables](/guides/tables) for interactive visualization.

These new callbacks:

* Adhere to Keras design philosophy.
* Reduce the cognitive load of using a single callback (`WandbCallback`) for everything.
* Make it easy for Keras users to modify the callback by subclassing it to support their niche use case.

## Track experiments with `WandbMetricsLogger`

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger` automatically logs Keras' `logs` dictionary that callback methods such as `on_epoch_end`, `on_batch_end` etc, take as an argument.

This tracks:

* Training and validation metrics defined in `model.compile`.
* System (CPU/GPU/TPU) metrics.
* Learning rate (both for a fixed value or a learning rate scheduler.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Initialize a new W&B run
wandb.init(config={"bs": 12})

# Pass the WandbMetricsLogger to model.fit
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` reference


| Parameter | Description | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch`, or an `int`): if `epoch`, logs metrics at the end of each epoch. If `batch`, logs metrics at the end of each batch. If an `int`, logs metrics at the end of that many batches. Defaults to `epoch`.                                 |
| `initial_global_step` | (int): Use this argument to correctly log the learning rate when you resume training from some initial_epoch, and a learning rate scheduler is used. This can be computed as step_size * initial_step. Defaults to 0. |

## Checkpoint a model using `WandbModelCheckpoint`

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

Use `WandbModelCheckpoint` callback to save the Keras model (`SavedModel` format) or model weights periodically and uploads them to W&B as a `wandb.Artifact` for model versioning. 

This callback is subclassed from [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) ,thus the checkpointing logic is taken care of by the parent callback.

This callback saves:

* The model that has achieved best performance based on the monitor.
* The model at the end of every epoch regardless of the performance.
* The model at the end of the epoch or after a fixed number of training batches.
* Only model weights or the whole model.
* The model either in `SavedModel` format or in `.h5` format.

Use this callback in conjunction with `WandbMetricsLogger`.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Initialize a new W&B run
wandb.init(config={"bs": 12})

# Pass the WandbModelCheckpoint to model.fit
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("models"),
    ],
)
```

### `WandbModelCheckpoint` reference

| Parameter | Description | 
| ------------------------- |  ---- | 
| `filepath`   | (str): path to save the mode file.|  
| `monitor`                 | (str): The metric name to monitor.         |
| `verbose`                 | (int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.   |
| `save_best_only`          | (Boolean): if `save_best_only=True`, it only saves the latest model or the model it considers the best, according to the defined by the `monitor` and `mode` attributes.   |
| `save_weights_only`       | (Boolean): if True, saves only the model's weights.                                            |
| `mode`                    | (`auto`, `min`, or `max`): For `val_acc`, set it to `max`, for `val_loss`, set it to `min`, and so on  |                     |
| `save_freq`               | ("epoch" or int): When using ‘epoch’, the callback saves the model after each epoch. When using an integer, the callback saves the model at end of this many batches. Note that when monitoring validation metrics such as `val_acc` or `val_loss`, `save_freq` must be set to "epoch" as those metrics are only available at the end of an epoch. |
| `options`                 | (str): Optional `tf.train.CheckpointOptions` object if `save_weights_only` is true or optional `tf.saved_model.SaveOptions` object if `save_weights_only` is false.    |
| `initial_value_threshold` | (float): Floating point initial "best" value of the metric to be monitored.       |

### Log checkpoints after N epochs

By default (`save_freq="epoch"`), the callback creates a checkpoint and uploads it as an artifact after each epoch. To create a checkpoint after a specific number of batches, set `save_freq` to an integer. To checkpoint after `N` epochs, compute the cardinality of the `train` dataloader and pass it to `save_freq`:

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### Efficiently log checkpoints on a TPU architecture

While checkpointing on TPUs you might encounter `UnimplementedError: File system scheme '[local]' not implemented` error message. This happens because the model directory (`filepath`) must use a cloud storage bucket path (`gs://bucket-name/...`), and this bucket must be accessible from the TPU server. We can however, use the local path for checkpointing which in turn is uploaded as an Artifacts.

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## Visualize model predictions using `WandbEvalCallback`

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

The `WandbEvalCallback` is an abstract base class to build Keras callbacks primarily for model prediction and, secondarily, dataset visualization.

This abstract callback is agnostic with respect to the dataset and the task. To use this, inherit from this base `WandbEvalCallback` callback class and implement the `add_ground_truth` and `add_model_prediction` methods.

The `WandbEvalCallback` is a utility class that provides methods to:

* Create data and prediction `wandb.Table` instances.
* Log data and prediction Tables as `wandb.Artifact`.
* Log the data table `on_train_begin`.
* log the prediction table `on_epoch_end`.

The following example uses `WandbClfEvalCallback` for an image classification task. This example callback logs the validation data (`data_table`) to W&B, performs inference, and logs the prediction (`pred_table`) to W&B at the end of every epoch.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


# Implement your model prediction visualization callback
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch, logs=None):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )


# ...

# Initialize a new W&B run
wandb.init(config={"hyper": "parameter"})

# Add the Callbacks to Model.fit
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbClfEvalCallback(
            validation_data=(X_test, y_test),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        ),
    ],
)
```

{{% alert %}}
The W&B [Artifact page](/guides/artifacts/explore-and-traverse-an-artifact-graph) includes Table logs by default, rather than the **Workspace** page.
{{% /alert %}}

### `WandbEvalCallback` reference

| Parameter            | Description                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) List of column names for the `data_table` |
| `pred_table_columns` | (list) List of column names for the `pred_table` |

### Memory footprint details

We log the `data_table` to W&B when the `on_train_begin` method is invoked. Once it's uploaded as a W&B Artifact, we get a reference to this table which can be accessed using `data_table_ref` class variable. The `data_table_ref` is a 2D list that can be indexed like `self.data_table_ref[idx][n]`, where `idx` is the row number while `n` is the column number. Let's see the usage in the example below.

### Customize the callback

You can override the `on_train_begin` or `on_epoch_end` methods to have more fine-grained control. If you want to log the samples after `N` batches, you can implement `on_train_batch_end` method.

{{% alert %}}
💡 If you are implementing a callback for model prediction visualization by inheriting `WandbEvalCallback` and something needs to be clarified or fixed, please let us know by opening an [issue](https://github.com/wandb/wandb/issues).
{{% /alert %}}

## `WandbCallback` [legacy]

Use the W&B library [`WandbCallback`](/ref/python/integrations/keras/wandbcallback) Class to automatically save all the metrics and the loss values tracked in `model.fit`.

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # code to set up your model in Keras

# Pass the callback to model.fit
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

You can watch the short video [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M).

For a more detailed video, watch [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases). You can review the [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb).

{{% alert %}}
See our [example repo](https://github.com/wandb/examples) for scripts, including a [Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py) and the [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) it generates.
{{% /alert %}}

The `WandbCallback` class supports a wide variety of logging configuration options: specifying a metric to monitor, tracking of weights and gradients, logging of predictions on training_data and validation_data, and more.

Check out [the reference documentation for the `keras.WandbCallback`](../../ref/python/integrations/keras/wandbcallback.md) for full details.

The `WandbCallback` 

* Automatically logs history data from any metrics collected by Keras: loss and anything passed into `keras_model.compile()`.
* Sets summary metrics for the run associated with the "best" training step, as defined by the `monitor` and `mode` attributes. This defaults to the epoch with the minimum `val_loss`. `WandbCallback` by default saves the model associated with the best `epoch`.
* Optionally logs gradient and parameter histogram.
* Optionally saves training and validation data for wandb to visualize.

### `WandbCallback` reference

| Arguments                  |                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) name of metric to monitor. Defaults to `val_loss`.                                                                   |
| `mode`                     | (str) one of {`auto`, `min`, `max`}. `min` - save model when monitor is minimized `max` - save model when monitor is maximized `auto` - try to guess when to save the model (default).                                                                                                                                                |
| `save_model`               | True - save a model when monitor beats all previous epochs False - don't save models                                       |
| `save_graph`               | (boolean) if True save model graph to wandb (default to True).                                                           |
| `save_weights_only`        | (boolean) if True, saves only the model's weights(`model.save_weights(filepath)`). Otherwise, saves the full model).   |
| `log_weights`              | (boolean) if True save histograms of the model's layer's weights.                                                |
| `log_gradients`            | (boolean) if True log histograms of the training gradients                                                       |
| `training_data`            | (tuple) Same format `(X,y)` as passed to `model.fit`. This is needed for calculating gradients - this is mandatory if `log_gradients` is `True`.       |
| `validation_data`          | (tuple) Same format `(X,y)` as passed to `model.fit`. A set of data for wandb to visualize. If you set this field, every epoch, wandb makes a small number of predictions and saves the results for later visualization.          |
| `generator`                | (generator) a generator that returns validation data for wandb to visualize. This generator should return tuples `(X,y)`. Either `validate_data` or generator should be set for wandb to visualize specific data examples.     |
| `validation_steps`         | (int) if `validation_data` is a generator, how many steps to run the generator for the full validation set.       |
| `labels`                   | (list) If you are visualizing your data with wandb this list of labels converts numeric output to understandable string if you are building a classifier with multiple classes. For a binary classifier, you can pass in a list of two labels \[`label for false`, `label for true`]. If `validate_data` and `generator` are both false, this does nothing.    |
| `predictions`              | (int) the number of predictions to make for visualization each epoch, max is 100.    |
| `input_type`               | (string) type of the model input to help visualization. can be one of: (`image`, `images`, `segmentation_mask`).  |
| `output_type`              | (string) type of the model output to help visualziation. can be one of: (`image`, `images`, `segmentation_mask`).    |
| `log_evaluation`           | (boolean) if True, save a Table containing validation data and the model's predictions at each epoch. See `validation_indexes`, `validation_row_processor`, and `output_row_processor` for additional details.     |
| `class_colors`             | (\[float, float, float]) if the input or output is a segmentation mask, an array containing an rgb tuple (range 0-1) for each class.                  |
| `log_batch_frequency`      | (integer) if None, callback logs every epoch. If set to integer, callback logs training metrics every `log_batch_frequency` batches.          |
| `log_best_prefix`          | (string) if None, saves no extra summary metrics. If set to a string, prepends the monitored metric and epoch with the prefix and saves the results as summary metrics.   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) an ordered list of index keys to associate with each validation example. If `log_evaluation` is True and you provide `validation_indexes`, does not create a Table of validation data. Instead, associates each prediction with the row represented by the `TableLinkMixin`. To obtain a list of row keys, use `Table.get_index() `.        |
| `validation_row_processor` | (Callable) a function to apply to the validation data, commonly used to visualize the data. The function receives an `ndx` (int) and a `row` (dict). If your model has a single input, then `row["input"]` contains the input data for the row. Otherwise, it contains the names of the input slots. If your fit function takes a single target, then `row["target"]` contains the target data for the row. Otherwise, it contains the names of the output slots. For example, if your input data is a single array, to visualize the data as an Image, provide `lambda ndx, row: {"img": wandb.Image(row["input"])}` as the processor. Ignored if `log_evaluation` is False or `validation_indexes` are present. |
| `output_row_processor`     | (Callable) same as `validation_row_processor`, but applied to the model's output. `row["output"]` contains the results of the model output.          |
| `infer_missing_processors` | (Boolean) Determines whether to infer `validation_row_processor` and `output_row_processor` if they are missing. Defaults to True. If you provide `labels`, W&B attempts to infer classification-type processors where appropriate.      |
| `log_evaluation_frequency` | (int) Determines how often to log evaluation results. Defaults to `0` to log only at the end of training. Set to 1 to log every epoch, 2 to log every other epoch, and so on. Has no effect when `log_evaluation` is False.    |

## Frequently Asked Questions

### How do I use `Keras` multiprocessing with `wandb`?

When setting `use_multiprocessing=True`, this error may occur:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

To work around it:

1. In the `Sequence` class construction, add: `wandb.init(group='...')`.
2. In `main`, make sure you're using `if __name__ == "__main__":` and put the rest of your script logic inside it.