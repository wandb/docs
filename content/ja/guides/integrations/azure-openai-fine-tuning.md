---
title: Azure OpenAI ファインチューニング
description: W&B を使って Azure OpenAI モデルをファインチューンする方法
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
Microsoft Azure 上で GPT-3.5 または GPT-4 モデルをファインチューニングする際、W&B を利用することでメトリクスの自動記録、体系的な評価が可能となり、W&B の実験管理や評価ツールを通じてモデルのパフォーマンスを分析・改善できます。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="Azure OpenAI ファインチューニング メトリクス" >}}

## 前提条件
- [公式 Azure ドキュメント](https://wandb.me/aoai-wb-int) に従い、Azure OpenAI サービスをセットアップしてください。
- APIキーを用いて W&B アカウントを設定してください。

## ワークフロー概要

### 1. ファインチューニングのセットアップ
- Azure OpenAI の要件に従い、トレーニングデータを準備します。
- Azure OpenAI でファインチューニング ジョブを設定します。
- W&B がファインチューニングのプロセスを自動的にトラッキングし、メトリクスやハイパーパラメーターをログに記録します。

### 2. 実験管理
ファインチューニング中、W&B は以下の内容を記録します:
- トレーニングおよびバリデーションのメトリクス
- モデルのハイパーパラメーター
- リソースの使用状況
- トレーニング Artifacts

### 3. モデルの評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使って以下を行います:
- モデル出力をリファレンスデータセットと比較評価
- 異なるファインチューニング run 間でのパフォーマンス比較
- 特定のテストケースにおけるモデルの振る舞い分析
- データに基づいてモデル選択の意思決定を行う

## 実際の事例
* [医療ノート生成デモ](https://wandb.me/aoai-ft-colab) をご覧いただくと、このインテグレーションを利用して以下を実現できることがわかります：
  - ファインチューニング実験の体系的トラッキング
  - ドメイン特化メトリクスを用いたモデル評価
* [ノートブックでのファインチューニングを体験できるインタラクティブなデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb) にもぜひ取り組んでみてください

## 追加リソース
- [Azure OpenAI W&B インテグレーション ガイド](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニング ドキュメント](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)