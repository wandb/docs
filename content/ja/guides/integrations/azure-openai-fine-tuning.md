---
title: Azure OpenAI ファインチューニング
description: W&B を使って Azure OpenAI モデルをファインチューンする方法
menu:
  default:
    identifier: azure-openai-fine-tuning
    parent: integrations
weight: 20
---

## イントロダクション
Microsoft Azure 上で GPT-3.5 または GPT-4 モデルをファインチューニングする際、W&B はメトリクスを自動で記録し、W&B の実験管理と評価ツールを通じて体系的にモデル性能の分析と改善をサポートします。

{{< img src="/images/integrations/aoai_ft_plot.png" alt="Azure OpenAI ファインチューニングメトリクス" >}}

## 前提条件
- [公式 Azure ドキュメント](https://wandb.me/aoai-wb-int) に従い Azure OpenAI サービスをセットアップする
- W&B アカウントを作成し、APIキー を設定する

## ワークフロー概要

### 1. ファインチューニング準備
- Azure OpenAI の要件に従ってトレーニングデータを準備する
- Azure OpenAI でファインチューニングジョブを設定する
- W&B がファインチューニングの進行を自動で管理し、メトリクスやハイパーパラメータを記録します

### 2. 実験管理
ファインチューニング中、W&B は以下を記録します:
- トレーニングとバリデーションのメトリクス
- モデルのハイパーパラメータ
- リソース使用状況
- トレーニング Artifacts

### 3. モデルの評価
ファインチューニング後、[W&B Weave](https://weave-docs.wandb.ai) を使って以下を行えます:
- モデルの出力をリファレンスデータセットと比較して評価
- 異なるファインチューニング run 間で性能を比較
- 特定のテストケースにおけるモデルの振る舞いを分析
- モデル選択のためのデータ活用型意思決定

## 実際の例
* [医療ノート生成デモ](https://wandb.me/aoai-ft-colab) を見て、このインテグレーションが以下のようなことをどう実現するのか確認できます:
  - ファインチューニング Experiments の体系的なトラッキング
  - ドメイン固有のメトリクスを用いたモデル評価
* [ノートブックでのファインチューニングのインタラクティブなデモ](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb) もお試しください

## 追加リソース
- [Azure OpenAI W&B インテグレーションガイド](https://wandb.me/aoai-wb-int)
- [Azure OpenAI ファインチューニング ドキュメント](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tuning?tabs=turbo%2Cpython&pivots=programming-language-python)