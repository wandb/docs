---
slug: /guides/integrations
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 統合

Weights & Biases統合によって、既存プロジェクト内で実験トラッキングとデータバージョン管理を迅速かつ簡単にセットアップできます。一般的なMLフレームワーク（例：[PyTorch](pytorch.md)）、ライブラリ（例：[Hugging Face](huggingface.md)）、またはサービス（例：[SageMaker](other/sagemaker.md))を使っている場合、以下と左側のナビゲーションバーにある統合を確認してください。

### 関連リンク​

* [例](https://github.com/wandb/examples): 弊社のすべての統合の、動作中のエンドツーエンドのGoogle Colabsとスクリプト例
* [動画チュートリアル](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk): PyTorch、Keras、およびその他のサービス向けのYouTube動画を視聴して、W&Bの使い方を学びましょう。


## 特定の統合向けガイド​

<Tabs
  defaultValue="frameworks"
  values={[
    {label: '一般的なMLフレームワーク', value: 'frameworks'},
    {label: '一般的なMLライブラリ', value: 'repositories'},
    {label: '一般的なツール', value: 'tools'},
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