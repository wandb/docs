---
title: チュートリアル
description: インタラクティブなチュートリアルで W&B の使い方を始めましょう。
menu:
  tutorials:
    identifier: tutorials
type: docs
cascade:
  type: docs
  menu:
    tutorials:
      parent: tutorials
no_list: true
---

## 基本

以下のチュートリアルでは、機械学習の実験管理、モデルの評価、ハイパーパラメータチューニング、モデルおよびデータセットのバージョン管理など、W&B の基本を学ぶことができます。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/experiments/">
      <h2 className="card-title">Experiments を記録する</h2>
    </a>
    <p className="card-content">W&B を使って機械学習の実験管理やモデルのチェックポイント作成、チームでのコラボレーションなどが行えます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/tables/">
      <h2 className="card-title">予測を可視化する</h2>
    </a>
    <p className="card-content">PyTorch を使って MNIST データ上で、モデルの予測をトレーニング中に記録・可視化・比較します。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/sweeps/">
      <h2 className="card-title">ハイパーパラメータをチューニングする</h2>
    </a>
    <p className="card-content">W&B Sweeps を使えば、学習率やバッチサイズ、隠れ層の数など、さまざまなハイパーパラメーターの組み合わせを自動で探索・整理できます。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/artifacts/">
      <h2 className="card-title">モデルやデータセットを管理する</h2>
    </a>
    <p className="card-content">W&B Artifacts を用いて ML 実験のパイプラインを管理しましょう。</p>
  {{< /card >}}
{{< /cardpane >}}

## 人気の ML フレームワークのチュートリアル

以下のチュートリアルでは、人気の ML フレームワークやライブラリを W&B で使う方法をステップバイステップで紹介しています。

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/pytorch">
      <h2 className="card-title">PyTorch</h2>
    </a>
    <p className="card-content">W&B を PyTorch コードへ統合し、パイプラインに実験管理機能を追加しましょう。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/huggingface">
      <h2 className="card-title">HuggingFace Transformers</h2>
    </a>
    <p className="card-content">Hugging Face モデルのパフォーマンスを、W&B インテグレーションで素早く可視化できます。</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/tensorflow">
      <h2 className="card-title">Keras</h2>
    </a>
    <p className="card-content">W&B と Keras を活用して、機械学習の実験管理やデータセットのバージョン管理、プロジェクトでのコラボレーションが可能です。</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/xgboost_sweeps/">
      <h2 className="card-title">XGBoost</h2>
    </a>
    <p className="card-content">W&B と XGBoost で機械学習の実験管理やデータセットのバージョン管理、プロジェクトでのコラボレーションができます。</p>
  {{< /card >}}
{{< /cardpane >}}

## その他のリソース

W&B AI Academy では、LLM のトレーニング・ファインチューニングやアプリケーションでの利用方法、MLOps や LLMOps ソリューションの実装が学べます。W&B のコースで実践的な ML チャレンジにも挑戦できます。

- Large Language Models (LLMs)
    - [LLM エンジニアリング: 構造化出力](https://www.wandb.courses/courses/steering-language-models?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [LLM搭載アプリの構築](https://www.wandb.courses/courses/building-llm-powered-apps?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [大規模言語モデルのトレーニングとファインチューニング](https://www.wandb.courses/courses/training-fine-tuning-LLMs?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- Effective MLOps
    - [モデルCI/CD](https://www.wandb.courses/courses/enterprise-model-management?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [効果的なMLOps: モデル開発](https://www.wandb.courses/courses/effective-mlops-model-development?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [機械学習のCI/CD (GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [プロダクションML パイプラインのデータ検証](https://www.wandb.courses/courses/data-validation-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [ビジネス意思決定の最適化のための機械学習](https://www.wandb.courses/courses/decision-optimization?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- W&B Models
    - [W&B 101](https://wandb.ai/site/courses/101/?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [W&B 201: モデルレジストリ](https://www.wandb.courses/courses/201-model-registry?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)