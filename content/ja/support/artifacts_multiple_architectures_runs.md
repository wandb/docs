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

モデルのバージョン管理には様々な方法があります。Artifacts は、特定のニーズに合わせたモデルのバージョン管理ツールを提供します。複数のモデルアーキテクチャーを探索するプロジェクトでは、アーティファクトをアーキテクチャーごとに分離するのが一般的なアプローチです。以下のステップを考慮してください。

1. 各異なるモデルアーキテクチャーに対して新しいアーティファクトを作成します。アーティファクトの `metadata` 属性を使用して、アーキテクチャーの詳細な説明を提供します。これは、run の `config` を使用する方法に似ています。
2. 各モデルに対して定期的にチェックポイントを `log_artifact` でログに記録します。W&B はこれらのチェックポイントの履歴を構築し、最新のものに `latest` エイリアスをラベル付けします。`architecture-name:latest` を使用して、任意のモデルアーキテクチャーの最新のチェックポイントを参照してください。