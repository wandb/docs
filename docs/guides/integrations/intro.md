---
slug: /guides/integrations
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Integrations

Weights & Biases integrations make it fast and easy to set up experiment tracking and data versioning inside existing projects. If you're using a popular ML framework (ex. [PyTorch](pytorch.md)), library (ex. [Hugging Face](huggingface.md)), or service (ex. [SageMaker](other/sagemaker.md)), check out the integrations below and in the navigation bar on the left!

### Related Links

* [Examples](https://github.com/wandb/examples): Working, end-to-end Google Colabs and script examples for all of our integrations
* [Video Tutorials](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): Learn to use W&B with YouTube videos for PyTorch, Keras, and more.

## Guides for Specific Integrations

<Tabs
  defaultValue="frameworks"
  values={[
    {label: 'Popular ML Frameworks', value: 'frameworks'},
    {label: 'Popular ML Libraries', value: 'repositories'},
    {label: 'Popular Tools', value: 'tools'},
  ]}>
  <TabItem value="frameworks">

* [Keras](keras.md)
* [PyTorch](pytorch.md)
* [PyTorch Lightning](lightning.md)
* [PyTorch Ignite](other/ignite.md)
* [TensorFlow](tensorflow.md)
* [Fastai](fastai/README.md)
* [Scikit-Learn](scikit.md)


  </TabItem>
  <TabItem value="repositories">

* [Hugging Face](huggingface.md)
* [PyTorch Geometric](pytorch-geometric.md)
* [spaCy](spacy.md)
* [YOLOv5](yolov5.md)
* [Simple Transformers](other/simpletransformers.md)
* [spaCy](spacy.md)
* [Catalyst](other/catalyst.md)
* [XGBoost](xgboost.md)
* [LightGBM](lightgbm.md)


  </TabItem>
  <TabItem value="tools">

* [TensorBoard](tensorboard.md)
* [SageMaker](other/sagemaker.md)
* [Kubeflow Pipelines](other/kubeflow-pipelines-kfp.md)
* [Dagster](./dagster.md)
* [Docker](other/docker.md)
* [Databricks](other/databricks.md)
* [Ray Tune](other/ray-tune.md)
* [OpenAI Gym](other/openai-gym.md)


  </TabItem>
</Tabs>
