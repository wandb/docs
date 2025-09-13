---
title: チュートリアル
description: インタラクティブなチュートリアルで W&B を使い始めましょう。
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

## 基礎

以下のチュートリアルでは、W&B による機械学習の実験管理、モデルの評価、ハイパーパラメータチューニング、モデルとデータセットのバージョン管理などの基礎を学べます。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/experiments/">
      <h2 className="card-title">実験をトラッキング</h2>
    </a>
    <p className="card-content">W&B を使って、機械学習の実験管理、モデルのチェックポイント保存、チームでのコラボレーションなどが行えます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/tables/">
      <h2 className="card-title">予測を可視化</h2>
    </a>
    <p className="card-content">PyTorch と MNIST データを使って、トレーニングの過程におけるモデルの予測を記録・可視化・比較します。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/sweeps/">
      <h2 className="card-title">ハイパーパラメーターをチューニング</h2>
    </a>
    <p className="card-content">W&B Sweeps を使うと、学習率、バッチサイズ、隠れ層数などのハイパーパラメーターの組み合わせを自動探索する仕組みを体系的に構築できます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/artifacts/">
      <h2 className="card-title">モデルとデータセットをトラッキング</h2>
    </a>
    <p className="card-content">W&B Artifacts を使って、機械学習の実験パイプラインをトラッキングしましょう。</p>
  {{< /card >}}
{{< /cardpane >}}


## 人気の ML フレームワーク向けチュートリアル
W&B と人気の ML フレームワーク／ライブラリを組み合わせる方法を、ステップごとに解説したチュートリアルです。以下をご覧ください。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/pytorch">
      <h2 className="card-title">PyTorch</h2>
    </a>
    <p className="card-content">PyTorch のコードに W&B を組み込み、パイプラインに実験管理を追加しましょう。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/huggingface">
      <h2 className="card-title">HuggingFace Transformers</h2>
    </a>
    <p className="card-content">W&B とのインテグレーションで、Hugging Face のモデルの性能を素早く可視化できます。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/tensorflow">
      <h2 className="card-title">Keras</h2>
    </a>
    <p className="card-content">W&B と Keras を使って、機械学習の実験管理、データセットのバージョン管理、プロジェクトでのコラボレーションを行いましょう。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/xgboost_sweeps/">
      <h2 className="card-title">XGBoost</h2>
    </a>
    <p className="card-content">W&B と XGBoost を使って、機械学習の実験管理、データセットのバージョン管理、プロジェクトでのコラボレーションを行いましょう。</p>
  {{< /card >}}
{{< /cardpane >}}

## その他のリソース

W&B AI Academy で、LLM の学習・ファインチューニング・アプリケーションへの活用方法を学びましょう。MLOps や LLMOps のソリューションを実装し、W&B のコースで実世界の ML 課題に取り組みます。

- 大規模言語モデル (LLMs)
    - [LLM エンジニアリング: 構造化出力](https://www.wandb.courses/courses/steering-language-models?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [LLM 駆動アプリの構築](https://www.wandb.courses/courses/building-llm-powered-apps?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [大規模言語モデルのトレーニングとファインチューニング](https://www.wandb.courses/courses/training-fine-tuning-LLMs?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- 効果的な MLOps
    - [モデルの CI/CD](https://www.wandb.courses/courses/enterprise-model-management?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [効果的な MLOps: モデル開発](https://www.wandb.courses/courses/effective-mlops-model-development?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [機械学習のための CI/CD (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [プロダクション ML パイプラインにおけるデータ検証](https://www.wandb.courses/courses/data-validation-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [ビジネス意思決定最適化のための機械学習](https://www.wandb.courses/courses/decision-optimization?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- W&B Models
    - [W&B 101](https://wandb.ai/site/courses/101/?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [W&B 201: モデルレジストリ](https://www.wandb.courses/courses/201-model-registry?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)