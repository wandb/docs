---
title: 複数のアーキテクチャーや Runs で Artifacts を使っていますか？
menu:
  support:
    identifier: ja-support-kb-articles-artifacts_multiple_architectures_runs
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

モデルのバージョン管理には様々な方法がありますが、Artifacts は特定のニーズに合わせたモデルのバージョン管理ツールを提供します。複数のモデルアーキテクチャーを検証するプロジェクトでは、アーキテクチャーごとに Artifact を分けて管理することが一般的です。以下の手順を参考にしてください。

1. 異なるモデルアーキテクチャーごとに新しい Artifact を作成します。Artifact の `metadata` 属性を使って、アーキテクチャーについて詳しく記述しましょう。これは run の `config` を使うイメージに近いです。
2. 各モデルについて、`log_artifact` で定期的にチェックポイントを記録します。W&B はこれらのチェックポイント履歴を自動的に管理し、最新のものには `latest` エイリアスが付与されます。特定のモデルアーキテクチャーの最新チェックポイントを参照したい場合は、`architecture-name:latest` のように指定します。