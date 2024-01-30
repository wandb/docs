---
description: 'Model Registry terms and concepts'
displayed_sidebar: ja
---

# Terms and concepts

<head>
  <title>Model Registry terms and concepts</title>
</head>

The following terms describe key components of the W&B Model Registry: [*model version*](#model-version), [*model artifact*](#model-artifact), and [*registered model*](#registered-model).

## Model version
A model version represents a single model checkpoint. Model versions are a snapshot at a point in time of a model and its files within an experiment. 

A model version is an immutable directory of data and metadata that describes a trained model. Add files that let you store (and restore) your model architecture and learned parameters. 

A model version belongs to one, and only one, [model artifact](#model-artifact). A model version can belong to zero or more, [registered models](#registered-model). Model versions are stored in a model artifact in the order they are logged to the model artifact. 


Store files within model versions that are produced from the serialization process provided by your modeling library (for example, [PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) and [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)).



<!-- [INSERT IMAGE] -->

## Model alias

Model aliases are mutable strings that allow you to uniquely identify or reference each version of your registered model used a semantically-related keyword. You can only assign an alias to one version of a registered model. This is because an alias should refer to a unique version when used programatically. It also allows aliases to be used to capture a model's state (champion, candidate, production).

It is common practice to use aliases such as  "best", "latest", "production", or "staging" to mark model versions with special purposes.

For example, suppose you create a model and assign it a `"best"` alias. You can refer to that specific model with `run.use_model` 

```python
import wandb
run = wandb.init()
name = f"{entity/project/model_artifact_name}:{alias}"
run.use_model(name=name)
```

## Model artifact
A model artifact is a group of logged [model versions](#model-version). Model versions are stored in a model artifact in the order they are logged to the model artifact. 

A model artifact can contain one or more model versions. A model artifact can be empty if no model versions are logged to it. 


For example, suppose you create a model artifact. During model training, you create model a version when you periodically save checkpoints. Each checkpoint will correspond to its own [model version](#model-version). All of the model versions belong to the model artifact you created at the beginning of your training script.

The proceeding image shows a model artifact that contains three model versions: v0, v1, and v2.

![](@site/static/images/models/mr1c.png)

View an [example model artifact here](https://wandb.ai/timssweeney/model\_management\_docs\_official\_v0/artifacts/model/mnist-zws7gt0n).

## Registered model
A registered model is a collection of pointers (links) to model versions. You can think of a registered model as a folder of "bookmarks". Each "bookmark" of a registered model is a pointer to a [model version](#model-version) that belongs to a [model artifact](#model-artifact). 

Registered models often represent candidate models for a single modeling use case or task. For example, you might create registered model for different image classification task based on the model you use: "ImageClassifier-ResNet50", "ImageClassifier-VGG16", "DogBreedClassifier-MobileNetV2" and so on.


View an [example Registered Model here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=versions).

