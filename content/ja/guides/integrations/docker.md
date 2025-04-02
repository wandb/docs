---
title: Docker
description: W&B と Docker を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker インテグレーション

W&B は、 コード が実行された Docker イメージ へのポインターを保存できます。これにより、以前の 実験 を実行された環境に正確に復元できます。 wandb ライブラリ は、この状態を永続化するために **WANDB_DOCKER** 環境変数 を探します。この状態を自動的に設定するいくつかのヘルパーを提供します。

### ローカル開発

`wandb docker` は、 dockerコンテナ を起動し、 wandb 環境変数 を渡し、 コード をマウントし、 wandb がインストールされていることを確認する コマンド です。デフォルトでは、この コマンド は TensorFlow、PyTorch、Keras、Jupyter がインストールされた Docker イメージ を使用します。同じ コマンド を使用して、独自の Docker イメージ を起動できます: `wandb docker my/image:latest`。この コマンド は、現在の ディレクトリー を コンテナ の "/app" ディレクトリー にマウントします。これは "--dir" フラグで変更できます。

### 本番環境

`wandb docker-run` コマンド は、 本番環境 の ワークロード 用に提供されています。これは `nvidia-docker` のドロップイン代替となることを意図しています。これは `docker run` コマンド へのシンプルなラッパーで、 認証情報 と **WANDB_DOCKER** 環境変数 を呼び出しに追加します。 "--runtime" フラグを渡さず、 `nvidia-docker` がマシンで利用可能な場合、これにより ランタイム が nvidia に設定されます。

### Kubernetes

Kubernetes で トレーニング の ワークロード を実行し、 k8s API が pod に公開されている場合（デフォルトの場合）。 wandb は、 Docker イメージ のダイジェストについて API にクエリを実行し、 **WANDB_DOCKER** 環境変数 を自動的に設定します。

## 復元

run が **WANDB_DOCKER** 環境変数 で計測されている場合、 `wandb restore username/project:run_id` を呼び出すと、 コード を復元する新しいブランチをチェックアウトし、 トレーニング に使用された正確な Docker イメージ を元の コマンド で事前に設定して 起動 します。
