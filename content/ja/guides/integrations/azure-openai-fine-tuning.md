---
title: Azure OpenAI のファインチューニング
description: W&B を使って Azure OpenAI モデルをファインチューンする方法
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
Microsoft Azure 上で GPT-3.5 や GPT-4 のモデルを W&B でファインチューニングすると、メトリクスを自動的に取得し、W&B の実験管理と評価ツールによる体系的な評価を通じて、モデル性能の追跡・分析・改善ができます。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="Azure OpenAI のファインチューニング メトリクス" >}}

## 前提条件
- [official Azure documentation](https://wandb.me/aoai-wb-int) に従って Azure OpenAI サービスをセットアップする。
- APIキー を使って W&B アカウントを設定する。

## ワークフローの概要

### 1. ファインチューニングのセットアップ
- Azure OpenAI の要件に従ってトレーニングデータを準備する。
- Azure OpenAI でファインチューニング ジョブを設定する。
- W&B がファインチューニングのプロセスを自動で追跡し、メトリクスやハイパーパラメーターをログします。

### 2. 実験管理
ファインチューニング中、W&B は次を記録します:
- トレーニングおよび検証のメトリクス
- モデルのハイパーパラメーター
- リソース使用状況
- トレーニングの Artifacts

### 3. モデルの評価
ファインチューニング後は、[W&B Weave](https://weave-docs.wandb.ai) を使って次を行います:
- 参照データセットに対するモデル出力の評価
- 異なるファインチューニングの Runs 間での性能比較
- 特定のテストケースにおけるモデルの振る舞いの分析
- モデル選択に関するデータに基づく意思決定

## 実例
* このインテグレーションが次をどのように支援するかを確認するため、[医療ノート生成のデモ](https://wandb.me/aoai-ft-colab) をご覧ください:
  - ファインチューニングの実験を体系的にトラッキング
  - ドメイン固有のメトリクスを用いたモデルの評価
* [ノートブックでのファインチューニングのインタラクティブなデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb) を試してみてください

## 追加リソース
- [Azure OpenAI W&B Integration Guide](https://wandb.me/aoai-wb-int)
- [Azure OpenAI Fine-tuning Documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)