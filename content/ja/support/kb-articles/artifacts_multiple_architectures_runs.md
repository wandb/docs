---
title: Artifacts を複数のアーキテクチャーおよび Runs で使用するには？
menu:
  support:
    identifier: ja-support-kb-articles-artifacts_multiple_architectures_runs
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

モデルをバージョン管理する方法はさまざまです。Artifacts は、ニーズに合わせたモデルのバージョン管理のためのツールを提供します。複数のモデル アーキテクチャーを検討するプロジェクトでは、アーキテクチャーごとにアーティファクトを分けるのが一般的です。次の手順を検討してください:

1. それぞれの異なるモデル アーキテクチャーごとに新しいアーティファクトを作成します。run での `config` の使い方と同様に、アーティファクトの `metadata` 属性を使ってアーキテクチャーの詳細な説明を付与します。
2. 各モデルについて、`log_artifact` で定期的にチェックポイントをログします。W&B はこれらのチェックポイントの履歴を構築し、最新のものに `latest` のエイリアスを付けます。`architecture-name:latest` を使って、任意のモデル アーキテクチャーの最新チェックポイントを参照できます。