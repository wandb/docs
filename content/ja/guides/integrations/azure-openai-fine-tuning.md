---
title: Azure OpenAI Fine-Tuning
description: W&B を使用して Azure OpenAI モデル をファインチューンする方法。
menu:
  default:
    identifier: ja-guides-integrations-azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
GPT-3.5 や GPT-4 モデルを Microsoft Azure 上でファインチューニングする際、W&B はメトリクスを自動的にキャプチャし、W&B の実験管理と評価ツールを通じて体系的な評価を促進することで、モデルのパフォーマンスを追跡、分析、向上させます。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="" >}}

## 前提条件
- [公式Azureドキュメント](https://wandb.me/aoai-wb-int)に従って Azure OpenAI サービスをセットアップします。
- API キーを使用して W&B アカウントを設定します。

## ワークフローの概要

### 1. ファインチューニングのセットアップ
- Azure OpenAI の要件に従ってトレーニングデータを準備します。
- Azure OpenAI でファインチューニングジョブを設定します。
- W&B はファインチューニングプロセスを自動的に追跡し、メトリクスとハイパーパラメーターをログします。

### 2. 実験管理
ファインチューニング中、W&B は以下をキャプチャします：
- トレーニングと検証のメトリクス
- モデルのハイパーパラメーター
- リソースの利用状況
- トレーニングアーティファクト

### 3. モデルの評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使用して：
- モデル出力をリファレンスデータセットと比較評価
- 異なるファインチューニング run 間でのパフォーマンスを比較
- 特定のテストケースでのモデルの振る舞いを分析
- モデル選択のためのデータ主導の意思決定を行う

## 実際の例
* [医療ノート生成デモ](https://wandb.me/aoai-ft-colab) を探索して、このインテグレーションが以下をどのように促進するか確認します：
  - ファインチューニング実験の体系的な追跡
  - ドメイン固有のメトリクスを使用したモデルの評価
* [ノートブックのファインチューニングに関するインタラクティブデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb)を進めます

## 追加リソース
- [Azure OpenAI W&B インテグレーションガイド](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニングドキュメント](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)