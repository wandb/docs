---
title: チュートリアル
description: Weights & Biases を使用するためのインタラクティブなチュートリアルを始めましょう。
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

次のチュートリアルでは、機械学習の実験管理、モデル評価、ハイパーパラメータチューニング、モデルとデータセットのバージョン管理などにおける W&B の基本を紹介します。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/experiments/">
      <h2 className="card-title">実験を追跡する</h2>
      <p className="card-content">W&B を使用して、機械学習の実験管理、モデルのチェックポイント、チームとのコラボレーションなどを行います。</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/tables/">
      <h2 className="card-title">予測を可視化する</h2>
      <p className="card-content">PyTorch を使用して、トレーニングの過程でモデル予測を追跡、可視化、比較します。データは MNIST を使用。</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/sweeps/">
      <h2 className="card-title">ハイパーパラメーターを調整する</h2>
      <p className="card-content">W&B Sweeps を使用して、学習率、バッチサイズ、隠れ層の数などのハイパーパラメーターの組み合わせを自動的に検索するための組織化された方法を作成します。</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/artifacts/">
      <h2 className="card-title">モデルとデータセットを追跡する</h2>
      <p className="card-content">W&B Artifacts を使用して、ML 実験パイプラインを追跡します。</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

## 人気のある ML フレームワークチュートリアル

以下のチュートリアルでは、W&B を使用して人気のある ML フレームワークやライブラリを使用するためのステップバイステップの情報を提供します。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/pytorch">
      <h2 className="card-title">PyTorch</h2>
      <p className="card-content">W&B を PyTorch のコードに統合して、実験管理をパイプラインに追加します。</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/huggingface">
      <h2 className="card-title">HuggingFace Transformers</h2>
      <p className="card-content">W&B インテグレーションを使って Hugging Face モデルのパフォーマンスをすばやく可視化します。</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/tensorflow">
      <h2 className="card-title">Keras</h2>
      <p className="card-content">W&B と Keras を使用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行います。</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/xgboost_sweeps/">
      <h2 className="card-title">XGBoost</h2>
      <p className="card-content">W&B と XGBoost を使用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行います。</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

## その他のリソース

W&B AI アカデミーで、アプリケーションでの LLM のトレーニング、ファインチューン、および使用方法を学ぶことができます。 MLOps および LLMOps ソリューションを実装します。 W&B コースを使用して、現実世界の ML 課題に取り組みましょう。

- 大規模言語モデル (LLMs)
    - [LLM エンジニアリング: 構造化出力](https://www.wandb.courses/courses/steering-language-models?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [LLM 搭載アプリの構築](https://www.wandb.courses/courses/building-llm-powered-apps?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [大規模言語モデルのトレーニングとファインチューニング](https://www.wandb.courses/courses/training-fine-tuning-LLMs?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- 効果的な MLOps
    - [モデル CI/CD](https://www.wandb.courses/courses/enterprise-model-management?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [効果的な MLOps: モデル開発](https://www.wandb.courses/courses/effective-mlops-model-development?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [機械学習の CI/CD (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [プロダクション ML パイプラインにおけるデータの検証](https://www.wandb.courses/courses/data-validation-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [ビジネス意思決定最適化のための機械学習](https://www.wandb.courses/courses/decision-optimization?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- W&B Models 
    - [W&B 101](https://wandb.ai/site/courses/101/?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [W&B 201: Model Registry](https://www.wandb.courses/courses/201-model-registry?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)