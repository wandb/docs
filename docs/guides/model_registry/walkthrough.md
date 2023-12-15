---
description: Learn how to use W&B for Model Management
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Walkthrough

The following guide shows you how to register and manage your ML models. By the end of this walkthrough you will:

* Log a Keras MNIST model to W&B Model Registry
* Download the MNIST model from the Model Registry
* ...[INSERT -  complete this bullet point after walkthrough is finalized]


:::note
* Code snippets that are not required to use the W&B Model Registry are hidden in collapsible cells. 
* Copy and paste the code snippets in the same order presented in this guide.
:::

## Setting up

Before you get started, install and import some Python dependencies:

<details>

<summary>Import Python modules</summary>

```python
import wandb

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
```

</details>


## Create a dataset artifact



<details>

<summary>Generate data</summary>

```python
def generate_raw_data(train_size=train_size):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    print("Generated {} rows of training data.".format(train_size))
    print("Generated {} rows of eval data.".format(eval_size))

    return (x_train[:train_size], y_train[:train_size]), (
        x_eval[:eval_size],
        y_eval[:eval_size],
    )


(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

</details>



<details>

<summary>Create dataset artifact</summary>

```python
entity = "<your-entity>"
project = "<project-name>"
job_type = "dataset_builder"

run = wandb.init(
    entity=entity,
    project=project,
    job_type=job_type,
    settings=wandb.Settings(start_method="fork"),
)

# Create W&B Table for training data
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# Create W&B Table for eval data
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})


artifact_name = model_use_case_id = "mnist_dataset"

# Create an artifact object
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# Add wandb.WBValue obj to the artifact.
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# Persist any changes made to the artifact.
artifact.save()

print("Published data to Artifact {}".format(artifact_name))

# Tell W&B this run is finished.
run.finish()
```


</details>


## Download dataset from artifact

Suppose at a later later date you want to use the dataset we uploaded. To do this, we download our dataset from W&B.



<details>

<summary>Download a dataset from artifact</summary>

```python
version = "latest"
job_type = "download_data"
name = "{}:{}".format("{}_ds".format(model_use_case_id), version)

run = wandb.init(entity=entity, project=project, job_type=job_type)

# Declare an artifact as an input to a run.
artifact = wandb.run.use_artifact(name)

print("Downlaoding Artifact {}".format(artifact.name))

# Get the WBValue object located at the artifact relative name.
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```


</details>



## Train a model



<details>

<summary>Training script</summary>

```python showLineNumbers
job_type = "train_model"

# Create a dictionary with hyperparameter values
config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

run = wandb.init(entity=entity, project=project, job_type=job_type, config=config)

# Store values from a config dictionary into variables for easy accessing
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
metrics = ["accuracy"]

optimizer = run.config["optimizer"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# Create model architecture
model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Generate labels for training data
y_train = keras.utils.to_categorical(y_train, num_classes)

# Create training and test set
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# Train the model
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# Save model locally with random title
path = "model.h5"
model.save(path)
```

</details>

## Log and link a model

```python
path = "./model.h5"
registered_model_name = "MNIST"  

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```


## Download a model to evaluate model performance
After training many models, you will likely want to evaluate the performance of those models. 

To evaluate a model version, you must first download a model version.

[TO DO]

```python
# Initialize a run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```


## Promote a model version 
Mark a model version ready for the next stage of your machine learning workflow with an *alias*. An alias is a [INSERT]. Each registered model can have one or more aliases. Each alias can only be assigned to a single version at a time.

For example, suppose that after evaluating a model's performance, you are confident that the model is ready for production. To promote that model version, add the `production` alias to that specific model version. 

You can add an alias to a model version interactively with the W&B App UI or programmatically with the Python SDK. The following steps show how to add an alias with the W&B App UI:

[TO DO - Insert steps]



:::tip
The `production` alias is one of the most common aliases we see used to mark a model as production-ready.
:::

 

