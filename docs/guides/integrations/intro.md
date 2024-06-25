---
slug: /guides/integrations
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# インテグレーション

Weights & Biasesのインテグレーションを使用すると、既存のプロジェクト内で実験管理やデータのバージョン管理を迅速かつ簡単に設定できます。人気のあるMLフレームワーク（例：[PyTorch](pytorch.md)）、ライブラリ（例：[Hugging Face](huggingface.md)）、またはサービス（例：[SageMaker](other/sagemaker.md)）を使用している場合は、以下および左側のナビゲーションバーのインテグレーションを確認してください！

### 関連リンク

* [Examples](https://github.com/wandb/examples): 各インテグレーションのためのノートブックおよびスクリプトの例を使用してコードを試してみてください
* [Video Tutorials](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): YouTubeの動画チュートリアルでW&Bの使い方を学びましょう

<iframe width="668" height="376" src="https://www.youtube.com/embed/hmewPDNUNJs?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk" title="Log Your First Run With W&amp;B" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## 特定のインテグレーション用ガイド

<Tabs
  defaultValue="frameworks"
  values={[
    {label: '人気のMLフレームワーク', value: 'frameworks'},
    {label: '人気のMLライブラリ', value: 'repositories'},
    {label: '人気のツール', value: 'tools'},
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
* [Ultralytics](ultralytics.md)
* [YOLOv5](yolov5.md)
* [Simple Transformers](other/simpletransformers.md)
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