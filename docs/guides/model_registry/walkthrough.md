---
description: Learn how to use W&B for Model Management
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Walkthrough

<head>
  <title>Walkthrough of how to use Model Management</title>
</head>

In this walkthrough you will learn how to use W&B for model management. More specifically, we cover how to track, visualize, and report on a complete production model workflow.

1. [Create a new Registered Model](#1-create-a-new-registered-model)
2. [Train & log Model Versions](#2-train-and-log-model-versions)
3. [Link Model Versions to the Registered Model](#3-link-model-versions-to-the-registered-model)
4. [Using a Model Version](#4-use-a-model-version)
5. [Evaluate Model Performance](#5-evaluate-model-performance)
6. [Promote a Version to Production](#6-promote-a-version-to-production)
7. [Use the Production Model for Inference](#7-consume-the-production-model)
8. [Build a Reporting Dashboard](#8-build-a-reporting-dashboard)


![](/images/models/models_landing_page.png)



## 1. Create a new registered model

First, create a registered model to hold all the candidate models for your modeling task. In this guide, we use [MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) as input 28 X 28 images with output classes from 0-9. 

The following tabs describe how to create a registered model interactively with the W&B App (Model registry and Artifact browser, respectively). The "Python SDK" tab describes how to programmatically create a registered model with W&B Python SDK.

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact Browser', value: 'browser'},
    {label: 'Python SDK', value: 'programmatic'},
  ]}>
  <TabItem value="registry">

1. Navigate to your model registry at [wandb.ai/registry/model](https://wandb.ai/registry/model).

![](/images/models/create_registered_model_1.png)


2. Click the **New registered model** button at the top of the Model Registry page.

![](/images/models/create_registered_model_3.png)

3. Select the entity the registered model will belong to from the **Owning Entity** dropdown.
4. Provide a name for your model in the **Model Name** field. 


  </TabItem>
  <TabItem value="browser">

1. Visit your Project's Artifact Browser: `wandb.ai/<entity>/<project>/artifacts`
2. Click the `+` icon on the bottom of the Artifact Browser Sidebar
3. Select `Type: model`, `Style: Collection`, and enter a name. In our case `MNIST Grayscale 28x28`. Remember, a Collection should map to a modeling task - enter a unique name that describes the use case.

![](/images/models/browser.gif)
  </TabItem>
    <TabItem value="programmatic">

If you already have a logged model version, you can link directly to a registered model from the SDK. If the registered model you specify doesn't exist, we will create it for you.

While manual linking is useful for one-off Models, it is often useful to programmatically link Model Versions to a Collection - consider a nightly job or CI pipeline that wants to link the best Model Version from every training job. Depending on your context and use case, you may use one of 3 different linking APIs:

**Fetch Model Artifact from Public API:**

```python
import wandb

# Fetch the Model Version via API
art = wandb.Api().artifact(...)
# Link the Model Version to the Model Collection
art.link("[[entity/]project/]collectionName")
```

**Model Artifact is "used" by the current Run:**

```python
import wandb

# Initialize a W&B run to start tracking
wandb.init()
# Obtain a reference to a Model Version
art = wandb.use_artifact(...)
# Link the Model Version to the Model Collection
art.link("[[entity/]project/]collectionName")
```

**Model Artifact is logged by the current Run:**

```python
import wandb

# Initialize a W&B run to start tracking
wandb.init()
# Create an Model Version
art = wandb.Artifact(...)
# Log the Model Version
wandb.log_artifact(art)
# Link the Model Version to the Collection
wandb.run.link_artifact(art, "[[entity/]project/]collectionName")
```
  </TabItem>
</Tabs>

## 2. Train and log model versions

Next, log a model from your training script:

1. (Optional) Declare your dataset as a dependency so that it is tracked for reproducibility and auditability.
2. **Serialize** your model to disk periodically (and/or at the end of training) using the serialization process provided by your modeling library (eg [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) & [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)).
3. **Add** your model files to an Artifact of type "model"
   * Note: We use the name `f'mnist-nn-{wandb.run.id}'`. While not required, it is advisable to name-space your "draft" Artifacts with the Run id in order to stay organized
4. (Optional) Log training metrics associated with the performance of your model during training.
   * Note: The data logged immediately before logging your Model Version will automatically be associated with that version
5. **Log** your model
   * Note: If you are logging multiple versions, it is advisable to add an alias of "best" to your Model Version when it outperforms the prior versions. This will make it easy to find the model with peak performance - especially when the tail end of training may overfit!

<Tabs
  defaultValue="withartifacts"
  values={[
    {label: 'Using Artifacts', value: 'withartifacts'},
    {label: 'Declare Dataset Dependency', value: 'datasetdependency'},
  ]}>
  <TabItem value="withartifacts">

```python
import wandb

# Always initialize a W&B run to start tracking
wandb.init()

# (Optional) Declare an upstream dataset dependency
# see the `Declare Dataset Dependency` tab for
# alternative examples.
dataset = wandb.use_artifact("mnist:latest")

# At the end of every epoch (or at the end of your script)...
# ... Serialize your model
model.save("path/to/model.pt")
# ... Create a Model Version
art = wandb.Artifact(f"mnist-nn-{wandb.run.id}", type="model")
# ... Add the serialized files
art.add_file("path/to/model.pt", "model.pt")
# (optional) Log training metrics
wandb.log({"train_loss": 0.345, "val_loss": 0.456})
# ... Log the Version
if model_is_best:
    # If the model is the best model so far,
    #  add "best" to the aliases
    wandb.log_artifact(art, aliases=["latest", "best"])
else:
    wandb.log_artifact(art)
```
  </TabItem>
  <TabItem value="datasetdependency">

If you would like to track your training data, you can declare a dependency by calling `wandb.use_artifact` on your dataset. Here are 3 examples of how you can declare a dataset dependency:

**Dataset stored in W&B**

```python
dataset = wandb.use_artifact("[[entity/]project/]name:alias")
```

**Dataset stored on Local Filesystem**

```python
art = wandb.Artifact("dataset_name", "dataset")
art.add_dir("path/to/data")  # or art.add_file("path/to/data.csv")
dataset = wandb.use_artifact(art)
```

**Dataset stored on Remote Bucket**

```python
art = wandb.Artifact("dataset_name", "dataset")
art.add_reference("s3://path/to/data")
dataset = wandb.use_artifact(art)
```
  </TabItem>
</Tabs>


After logging 1 or more Model Versions, you will notice that your will have a new Model Artifact in your Artifact Browser. Here, we can see the results of logging 5 versions to an artifact named `mnist_nn-1r9jjogr`.

![](/images/models/train_log_model_version_browser.png)

If you are following along the example notebook, you should see a Run Workspace with charts similar to the image below

![](/images/models/train_log_model_version_notebook.png)

## 3. Link model versions to the registered model

Link a model version to the registered model with the W&B App or programmatically with the Python SDK.

<Tabs
  defaultValue="manual_link"
  values={[
    {label: 'Manual Linking', value: 'manual_link'},
    {label: 'Programmatic Linking', value: 'program_link'},
  ]}>
  <TabItem value="manual_link">


1. Navigate to the Model Version of interest
2. Click the link icon
3. Select the target Registered Model
4. (optional): Add additional aliases


  </TabItem>
  <TabItem value="program_link">

The following code snippets demonstrate different linking API you can use to programmatically link a model version to a registered model:

**Fetch Model Artifact from Public API:**

```python
import wandb

# Fetch the Model Version via API
art = wandb.Api().artifact(...)

# Link the Model Version to the Model Collection
art.link("[[entity/]project/]collectionName")
```

**Model Artifact is "used" by the current Run:**

```python
import wandb

# Initialize a W&B run to start tracking
wandb.init()

# Obtain a reference to a Model Version
art = wandb.use_artifact(...)

# Link the Model Version to the Model Collection
art.link("[[entity/]project/]collectionName")
```

**Model Artifact is logged by the current Run:**

```python
import wandb

# Initialize a W&B run to start tracking
wandb.init()

# Create an Model Version
art = wandb.Artifact(...)

# Log the Model Version
wandb.log_artifact(art)

# Link the Model Version to the Collection
wandb.run.link_artifact(art, "[[entity/]project/]collectionName")
```
  </TabItem>
</Tabs>


After you link the model version, you will see hyperlinks that connect the version in the registered model to the source artifact. The artifact will also have hyperlinks that connect to the model version.

![](@site/static/images/models/train_log_model_version.png)


:::tip
This [companion colab notebook](http://wandb.me/models_quickstart) covers step 2-3 in the first code block and steps 4-6 in the second code block.
:::


## 4. Use a model version

Next, consume the model. For example, perhaps to you want to evaluate its performance, make predictions against a dataset, or use in a live production context. The following code snippet shows how to  use a model with the Python SDK:

```python
import wandb

# Always initialize a W&B run to start tracking
wandb.init()

# Download your Model Version files
path = wandb.use_artifact("[[entity/]project/]collectionName:latest").download()

# Reconstruct your model object in memory:
# `make_model_from_data` below represents your deserialization logic
# to load in a model from disk
model = make_model_from_data(path)
```

## 5. Evaluate model performance

After training many models, you will likely want to evaluate the performance of those models. In most circumstances you will have a test dataset that was not used for training or validation. To evaluate a model version, you will want to first complete step 4 above to load a model into memory. Then:

1. (Optional) Declare a data dependency to your evaluation data
2. Log metrics, media, tables, and anything else useful for evaluation

```python
# ... continuation from 4

# (Optional) Declare an upstream evaluation dataset dependency
dataset = wandb.use_artifact("mnist-evaluation:latest")

# Evaluate your model according to your use-case
loss, accuracy, predictions = evaluate_model(model, dataset)

# Log out metrics, images, tables, or any data useful for evaluation.
wandb.log({"loss": loss, "accuracy": accuracy, "predictions": predictions})
```

If you are executing similar code, as demonstrated in the notebook, you should see a workspace similar to the image below - here we even show model predictions against the test data!

![](/images/models/evaluate_model_performance.png)

## 6. Promote a version to production

Next, specify a model version to use for production with an alias. Each registered model can have one or more aliases. Each alias can only be assigned to a single Version at a time.

:::tip
The `production` alias is one of the most common aliases we see used to mark a model as production-ready.
:::

The following tabs demonstrate how to add an alias with the interactively with the W&B App and programmatically with the Python SDK:

<Tabs
  defaultValue="UI_interface"
  values={[
    {label: 'W&B App UI', value: 'UI_interface'},
    {label: 'Python SDK', value: 'api'},
  ]}>
  <TabItem value="UI_interface">

![](/images/models/promote_version_to_prod_1.png)
  </TabItem>
  <TabItem value="api">

Follow steps in [Part 3. Link Model Versions to the Collection](#3-link-model-versions-to-the-registered-model) and add the aliases you want to the `aliases` parameter.
  </TabItem>
</Tabs>

The image below shows the new `production` alias added to v1 of the Registered Model!

![](/images/models/promote_version_to_prod_2.png)

## 7. Consume the production model

Finally, use your production model for inference. See the [Use a model version](#4-use-a-model-version) for more information. In this example, we use the Python SDK:

```python
wandb.use_artifact("[[entity/]project/]registeredModelName:production")
```

You can reference a version within a registered model using different alias strategies:

* `latest` - which will fetch the most recently linked Version
* `v#` - using `v0`, `v1`, `v2`, ... you can fetch a specific version in the Registered Model
* `production` - you can use any custom alias that you and your team have assigned

## 8. Build a reporting dashboard

Using Weave Panels, you can display any of the Model Registry/Artifact views inside of Reports! See a [demo here](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/reports/MNIST-Grayscale-28x28-Model-Dashboard--VmlldzoyMDI0Mzc1). Below is a full-page screenshot of an example Model Dashboard.

![](/images/models/build_reporting_dashboard.png)
