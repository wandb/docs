---
title: Using artifacts with multiple architectures and runs?
menu:
  support:
    identifier: ja-support-kb-articles-artifacts_multiple_architectures_runs
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

モデルをバージョン管理するには、さまざまなメソッドがあります。Artifacts は、特定のニーズに合わせたモデルのバージョン管理ツールを提供します。複数のモデルのアーキテクチャーを調査するプロジェクトの一般的なアプローチとしては、Artifacts をアーキテクチャーごとに分離する方法があります。以下の手順を検討してください。

1. 異なるモデルのアーキテクチャーごとに新しい artifact を作成します。Artifacts の `metadata` 属性を使用して、run の `config` の使用と同様に、アーキテクチャーの詳細な説明を提供します。
2. 各モデルについて、`log_artifact` を使用してチェックポイントを定期的に ログ 記録します。W&B はこれらのチェックポイントの履歴を作成し、最新のチェックポイントに `latest` エイリアスを付けます。`architecture-name:latest` を使用して、任意のモデルのアーキテクチャーの最新のチェックポイントを参照します。
