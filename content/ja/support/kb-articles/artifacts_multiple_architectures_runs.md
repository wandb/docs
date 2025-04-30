---
title: 複数のアーキテクチャーと run で Artifacts を使用する方法は？
menu:
  support:
    identifier: ja-support-kb-articles-artifacts_multiple_architectures_runs
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

様々なモデルのバージョン管理方法があります。Artifacts は特定のニーズに合わせたモデルバージョン管理のためのツールを提供します。複数のモデル アーキテクチャーを探索するプロジェクトに共通のアプローチは、アーキテクチャーごとにアーティファクトを分けることです。次のステップを考慮してください。

1. 各異なるモデル アーキテクチャーに新しいアーティファクトを作成します。アーティファクトの `metadata` 属性を使用して、アーキテクチャーの詳細な説明を提供します。これは run の `config` の使用と似ています。
2. 各モデルについて、定期的にチェックポイントを `log_artifact` でログします。W&B はこれらのチェックポイントの履歴を構築し、最新のものには `latest` エイリアスを付けます。`architecture-name:latest` を使用して、任意のモデル アーキテクチャーの最新のチェックポイントを参照してください。