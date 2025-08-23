---
title: チュートリアル
description: インタラクティブなチュートリアルで W&B の使い方を始めましょう。
cascade:
  menu:
    tutorials:
      parent: tutorials
  type: docs
menu:
  tutorials:
    identifier: ja-tutorials-_index
no_list: true
type: docs
---

## 基本

以下のチュートリアルでは、W&B を使った機械学習実験管理、モデルの評価、ハイパーパラメータチューニング、モデルおよびデータセットのバージョン管理など、基本的な使い方を解説します。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/experiments/">
      <h2 className="card-title">Experiments を記録する</h2>
    </a>
    <p className="card-content">W&B を活用して機械学習の実験管理、モデルのチェックポイント作成、チームメンバーとのコラボレーションなどを行えます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/tables/">
      <h2 className="card-title">予測結果を可視化する</h2>
    </a>
    <p className="card-content">PyTorch で MNIST データを使いながら、トレーニング中のモデル予測を記録・可視化し比較できます。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/sweeps/">
      <h2 className="card-title">ハイパーパラメータをチューニングする</h2>
    </a>
    <p className="card-content">W&B Sweeps を使えば、学習率やバッチサイズ、隠れ層の数など様々なハイパーパラメーターの値の組み合わせを自動で探索できます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/artifacts/">
      <h2 className="card-title">Models と Datasets を管理する</h2>
    </a>
    <p className="card-content">W&B Artifacts を使って ML 実験パイプラインを記録・管理しましょう。</p>
  {{< /card >}}
{{< /cardpane >}}


## 人気の ML フレームワークのチュートリアル

以下のチュートリアルでは、主要な ML フレームワークやライブラリと W&B の連携方法をステップバイステップで紹介しています。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/pytorch">
      <h2 className="card-title">PyTorch</h2>
    </a>
    <p className="card-content">あなたの PyTorch コードと W&B を連携して、パイプラインに実験管理機能を追加できます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/huggingface">
      <h2 className="card-title">HuggingFace Transformers</h2>
    </a>
    <p className="card-content">W&B のインテグレーションを活用して、Hugging Face モデルのパフォーマンスをすぐに可視化しましょう。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/tensorflow">
      <h2 className="card-title">Keras</h2>
    </a>
    <p className="card-content">Keras と W&B を活用して、機械学習実験管理、データセットのバージョン管理、プロジェクトの共同作業が行えます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/xgboost_sweeps/">
      <h2 className="card-title">XGBoost</h2>
    </a>
    <p className="card-content">XGBoost と W&B を活用して、機械学習の実験管理やデータセットのバージョン管理、プロジェクト協働を効率化できます。</p>
  {{< /card >}}
{{< /cardpane >}}

## その他のリソース

W&B AI Academy では、LLM を使った学習・ファインチューニング・活用方法を学べます。MLOps や LLMOps の実践方法、実世界の課題に取り組むためのコースも用意しています。

- 大規模言語モデル（LLM）
    - [LLM Engineering: Structured Outputs](https://www.wandb.courses/courses/steering-language-models?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [Building LLM-Powered Apps](https://www.wandb.courses/courses/building-llm-powered-apps?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [Training and Fine-tuning Large Language Models](https://www.wandb.courses/courses/training-fine-tuning-LLMs?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- 実践 MLOps
    - [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [Effective MLOps: Model Development](https://www.wandb.courses/courses/effective-mlops-model-development?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [CI/CD for Machine Learning (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [本番 ML パイプラインのデータ検証](https://www.wandb.courses/courses/data-validation-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [ビジネス意思決定のための機械学習](https://www.wandb.courses/courses/decision-optimization?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- W&B Models
    - [W&B 101](https://wandb.ai/site/courses/101/?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [W&B 201: Model Registry](https://www.wandb.courses/courses/201-model-registry?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)