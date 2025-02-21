---
title: Using artifacts with multiple architectures and runs?
menu:
  support:
    identifier: ja-support-artifacts_multiple_architectures_runs
tags:
- artifacts
toc_hide: true
type: docs
---

モデルをバージョン管理するには、さまざまな メソッド があります。 Artifacts は、特定のニーズに合わせたモデルの バージョン管理 の ツール を提供します。複数のモデルの アーキテクチャー を調査する プロジェクト の一般的なアプローチとしては、 Artifacts を アーキテクチャー ごとに分離する方法があります。以下の手順を検討してください。

1. 個別のモデル アーキテクチャー ごとに新しい アーティファクト を作成します。 Artifacts の `metadata` 属性を使用して、run の `config` の使用と同様に、 アーキテクチャー の詳細な説明を提供します。
2. モデルごとに、`log_artifact` を使用して チェックポイント を定期的に ログ します。W&B はこれらの チェックポイント の履歴を作成し、最新の チェックポイント に `latest` エイリアス を付けてラベル付けします。`architecture-name:latest` を使用して、モデル アーキテクチャー の最新の チェックポイント を参照します。
