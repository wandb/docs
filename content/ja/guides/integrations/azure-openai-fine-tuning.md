---
title: Azure OpenAI ファインチューニング
description: Azure OpenAI モデルを Fine-Tune する方法とW&Bの使用方法。
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
Microsoft Azureを使用してGPT-3.5やGPT-4モデルをファインチューニングすることで、W&Bはメトリクスを自動的にキャプチャし、W&Bの実験管理および評価ツールを通じて系統的な評価を促進することで、モデルの性能を追跡し、分析し、改善します。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="" >}}

## 前提条件
- [公式のAzureドキュメント](https://wandb.me/aoai-wb-int)に従ってAzure OpenAIサービスをセットアップします。
- APIキーを使用してW&Bアカウントを設定します。

## ワークフローの概要

### 1. ファインチューニングのセットアップ
- Azure OpenAIの要件に従ってトレーニングデータを準備します。
- Azure OpenAIでファインチューニングジョブを設定します。
- W&Bはファインチューニングプロセスを自動的に追跡し、メトリクスとハイパーパラメーターをログします。

### 2. 実験管理
ファインチューニング中、W&Bは以下をキャプチャします：
- トレーニングと検証のメトリクス
- モデルのハイパーパラメーター
- リソースの利用状況
- トレーニングアーティファクト

### 3. モデルの評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使用して以下を行います：
- モデルの出力を参照データセットと比較評価します。
- 異なるファインチューニングのrun間で性能を比較します。
- 特定のテストケースでモデルの振る舞いを分析します。
- モデル選択のためのデータドリブンの意思決定を行います。

## 実際の例
* [医療メモ生成デモ](https://wandb.me/aoai-ft-colab)を探索して、このインテグレーションがどのように以下を実現するかをご覧ください：
  - ファインチューニング実験の体系的な追跡
  - ドメイン固有のメトリクスを使用したモデルの評価
* [ノートブックのファインチューニングのインタラクティブデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb)を体験してください。

## 追加リソース
- [Azure OpenAI W&B Integration Guide](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニングドキュメント](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)