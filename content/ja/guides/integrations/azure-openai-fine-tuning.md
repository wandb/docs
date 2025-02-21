---
title: Azure OpenAI Fine-Tuning
description: W&B を使用して Azure OpenAI モデル を ファインチューン する方法。
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
W&B を使用して Microsoft Azure 上で GPT-3.5 または GPT-4 モデルをファインチューニングすると、メトリクスを自動的にキャプチャし、W&B の 実験管理 および評価ツールを通じて体系的な評価を促進することで、モデルのパフォーマンスを追跡、分析、改善できます。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="" >}}

## 前提条件
- [Azure の公式ドキュメント](https://wandb.me/aoai-wb-int)に従って Azure OpenAI サービスをセットアップします。
- APIキー で W&B アカウントを設定します。

## ワークフローの概要

### 1. ファインチューニングのセットアップ
- Azure OpenAI の要件に従って、トレーニングデータを準備します。
- Azure OpenAI でファインチューニングジョブを設定します。
- W&B は、ファインチューニングの プロセス を自動的に追跡し、メトリクスと ハイパーパラメーター を ログ 記録します。

### 2. 実験管理
ファインチューニング中、W&B は以下をキャプチャします。
- トレーニング および 検証 メトリクス
- モデル の ハイパーパラメーター
- リソース の利用状況
- トレーニング Artifacts

### 3. モデル の評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使用して以下を行います。
- 参照 データセット に対して モデル の出力を評価する
- さまざまなファインチューニング Runs でのパフォーマンスを比較する
- 特定のテストケースについて モデル の 振る舞い を分析する
- データ に基づいた モデル 選択の意思決定を行う

## 実際の例
* [医療メモ生成 デモ](https://wandb.me/aoai-ft-colab) を調べて、この インテグレーション がどのように促進するかを確認してください。
  - ファインチューニング Experiments の体系的な追跡
  - ドメイン固有の メトリクス を使用した モデル の評価
* [ノートブックのファインチューニングのインタラクティブ デモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb) を見てみましょう

## 追加リソース
- [Azure OpenAI W&B インテグレーション ガイド](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニング ドキュメント](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)
