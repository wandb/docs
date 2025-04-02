---
title: Azure OpenAI Fine-Tuning
description: W&B を使用して Azure OpenAI モデルを ファインチューン する方法。
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
Microsoft Azure 上で GPT-3.5 または GPT-4 モデルをファインチューニングする際、W&B を使用することで、メトリクスの自動的なキャプチャや W&B の 実験管理 および評価 ツールによる体系的な評価が促進され、モデルのパフォーマンスを追跡、分析、改善できます。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="" >}}

## 前提条件
- [Azure の公式ドキュメント](https://wandb.me/aoai-wb-int) に従って Azure OpenAI サービスをセットアップします。
- APIキー で W&B アカウントを設定します。

## ワークフローの概要

### 1. ファインチューニングのセットアップ
- Azure OpenAI の要件に従ってトレーニングデータを準備します。
- Azure OpenAI でファインチューニングジョブを設定します。
- W&B は、ファインチューニングプロセスを自動的に追跡し、メトリクスとハイパーパラメータをログに記録します。

### 2. 実験管理
ファインチューニング中、W&B は以下をキャプチャします。
- トレーニング および 検証メトリクス
- モデル ハイパーパラメータ
- リソース使用率
- トレーニング Artifacts

### 3. モデルの評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使用して以下を行います。
- 参照データセットに対するモデル出力を評価します
- 異なるファインチューニング Runs 全体のパフォーマンスを比較します
- 特定のテストケースにおけるモデルの 振る舞い を分析します
- データに基づいたモデル選択の意思決定を行います

## 実際の例
* [医療記録生成 デモ](https://wandb.me/aoai-ft-colab) を見て、この インテグレーション がどのように促進するかを確認します。
  - ファインチューニング Experiments の体系的な追跡
  - ドメイン固有のメトリクスを使用した モデル評価
* [ノートブックのファインチューニングに関するインタラクティブなデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb) を試してみてください。

## 追加リソース
- [Azure OpenAI W&B インテグレーション ガイド](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニング ドキュメント](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
