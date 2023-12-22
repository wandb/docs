---
description: 'Model Registry terms and concepts'
displayed_sidebar: default
---

# Terms and concepts

<head>
  <title>Model Registry terms and concepts</title>
</head>

The following terms are used to describe the W&B Model Registry: [*model version*](#model-version), [*model artifact*](#model-artifact), and [*registered model*](#registered-model).

## Model version
A model version is an immutable directory of data and metadata that describes a trained model. Add files that let you store (and restore) your model architecture and learned parameters.

Store files within model versions that are produced from the serialization process provided by your modeling library (for example, [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) and [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)).



A model version belongs to one, and only one, [model artifact](#model-artifact). A model version can belong to zero or more, [registered models](#registered-model).

<!-- [INSERT IMAGE] -->

## Model alias

[INSERT - to do]

It is common practice to use aliases such as  "best", "latest", "production", or "staging" to mark model versions with special purposes.

## Model artifact
A model artifact is a sequence of logged model versions. 

<!-- Model artifacts can alias specific versions so that downstream consumers can pin that alias.  -->

For example, suppose you create a model artifact. During model training, you create model a version when you periodically save checkpoints. Each checkpoint will correspond to its own [model version](#model-version). All of the model versions will belong to the model artifact you created at the beginning of your training script.

The proceeding image shows a model artifact that contains three model versions: v0, v1, and v2.

![](@site/static/images/models/mr1c.png)

View an [example model artifact](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n).

:::tip
Model artifacts often track the progress of training.
:::

## Registered model
A registered model is a collection of pointers (links) to model versions. You can think of a registered model as a folder of "bookmarks". Each "bookmark" of a registered model is a pointer to a [model version](#model-version) that belongs to a [model artifact](#model-artifact). 

Registered models often represent candidate models for a single modeling use case or task. For example, [INSERT EXAMPLE].


View an [example Registered Model](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/MNIST%20Grayscale%2028x28).

<!-- This image is confusing; we show a model collection, but don't specify it's a model artifact. -->
<!-- ![](/images/models/diagram_doc.png) -->



<!-- Move this to another page -->
<!-- The separation of artifacts and registered models allows you to produce a massive number of artifacts (think of them like "draft models"), and periodically _link_ your high performing versions to a the curated Registered Model. Then use aliases to mark which version in a registered model is at a certain stage in the lifecycle. Each person in your team can collaborate on a single use case, while having the freedom to explore and experiment without polluting namespaces or conflicting with others' work. -->



<!-- :::info
A model is an artifact with `type="model"`.  A model version is an artifact version that belongs to that artifact. A registered model is an artifact collection that has a type set to `model` (`type="model"`).
::: -->