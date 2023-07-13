---
slug: /guides/integrations
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# インテグレーション

Weights & Biasesのインテグレーションにより、既存のプロジェクト内での実験トラッキングとデータバージョニングの設定が簡単かつ迅速に行えます。もしあなたが一般的なMLフレームワーク（例：[PyTorch](pytorch.md)）、ライブラリ（例：[Hugging Face](huggingface.md)）、またはサービス（例：[SageMaker](other/sagemaker.md)）を使用している場合は、下記のインテグレーションや左側のナビゲーションバー内のものをチェックしてください！

### 関連リンク

* [Examples](https://github.com/wandb/examples)：すべてのインテグレーションに対応した、エンドツーエンドのGoogle Colabsとスクリプト例
* [ビデオチュートリアル](https://www.youtube.com/playlist?list=PLD80i8An1OEGajeVo15ohAQYF1Ttle0lk)：PyTorch、KerasなどのYouTubeビデオでW&Bを使いこなす方法を学ぶ

## 特定のインテグレーションのためのガイド

<Tabs
  defaultValue="frameworks"
  values={[
    {label: '人気のあるMLフレームワーク', value: 'frameworks'},
    {label: '人気のあるMLライブラリ', value: 'repositories'},
    {label: '人気のあるツール', value: 'tools'},
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