---
description: 'Model Management: Data Model & Terminology'
displayed_sidebar: default
---

# Concepts

<head>
  <title>Model Management Concepts</title>
</head>

The following are data model and terminology used to describe the W&B Model Registry:

* *Model version:* a package of data and metadata describing a trained model.
* *Model artifact:* a sequence of logged model versions. Model artifacts often track the progress of training.
* *Registered model:* a selection of linked model versions. Registered models often represent candidate models for a single modeling use case or task.

:::tip
A model version will always belong to one, and only one. model artifact. However, a model version can belong to zero or more Registered Models.
:::

:::info
A model is an artifact with `type="model"`.  A model version is an artifact version that belongs to that artifact. A registered model is an artifact collection that has a type set to `model` (`type="model"`).
:::

A model version is an immutable directory of data; it is up to you to decide what files and formats are appropriate to store (and restore) your model architecture and learned parameters. Typically you will want to store files that are produced from the serialization process provided by your modeling library (for example, [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) and [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)).

A model artifact is a sequence of model versions. Model artifacts can alias specific versions so that downstream consumers can pin that alias. It is common for a W&B run to produce many versions of a model while training (periodically saving checkpoints). Using this approach, each individual model trained by the run corresponds to its own model artifact. Each checkpoint corresponds to its own model version of the respective model artifact. 

View an [example model artifact](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n)

![](@site/static/images/models/mr1c.png)

A registered model is a set of links to model versions. A registered model can be accessed exactly like model artifacts (identified by `[[entityName/]/projectName]/registeredModelName:alias`), however it acts more like a folder of "bookmarks". Each "version" of a registered model is a link to a model version that belongs to a model artifact of the same type. A model version can be linked to more than one registered models. Typically you will create a registered model for each of your use cases and use aliases like "production" or "staging" to mark versions with special purposes. 

View an [example Registered Model](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/MNIST%20Grayscale%2028x28)

![](/images/models/diagram_doc.png)

While developing an ML model, you will likely have dozens, hundreds, or even thousands of runs that produce model versions. Most likely, not all of those models are great; often you are iterating on scripts, parameters, architectures, preprocessing logic and more. 

The separation of artifacts and registered models allows you to produce a massive number of artifacts (think of them like "draft models"), and periodically _link_ your high performing versions to a the curated Registered Model. Then use aliases to mark which version in a registered model is at a certain stage in the lifecycle. Each person in your team can collaborate on a single use case, while having the freedom to explore and experiment without polluting namespaces or conflicting with others' work.
