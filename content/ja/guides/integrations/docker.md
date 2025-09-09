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

W&B は、あなたのコードが実行された Docker イメージへの参照を保存でき、過去の実験を実行当時とまったく同じ環境に復元できます。wandb ライブラリは、この状態を保持するための **WANDB_DOCKER** 環境変数を探します。自動でこの状態を設定するためのヘルパーもいくつか提供しています。

### ローカル開発

`wandb docker` は、docker コンテナを起動し、wandb の環境変数を渡し、あなたのコードをマウントし、wandb がインストールされていることを保証するコマンドです。デフォルトでは、このコマンドは TensorFlow、PyTorch、Keras、Jupyter がインストールされた docker イメージを使用します。同じコマンドで自分の docker イメージも起動できます: `wandb docker my/image:latest`。このコマンドは現在のディレクトリーをコンテナの "/app" ディレクトリーにマウントします。これは "--dir" フラグで変更できます。

### プロダクション

`wandb docker-run` コマンドはプロダクションのワークロード向けに提供されています。これは `nvidia-docker` のドロップイン置き換えとして設計されています。`docker run` コマンドへのシンプルなラッパーで、あなたの認証情報と **WANDB_DOCKER** 環境変数を呼び出しに追加します。"--runtime" フラグを渡さず、かつマシンで `nvidia-docker` が利用可能な場合、ランタイムが nvidia に設定されることも保証します。

### Kubernetes

トレーニングのワークロードを Kubernetes で実行しており、k8s API があなたの pod に公開されている場合 \(デフォルトでそうです\)、wandb は docker イメージのダイジェストを取得するために API をクエリし、**WANDB_DOCKER** 環境変数を自動的に設定します。

## 復元

run が **WANDB_DOCKER** 環境変数で設定されている場合、`wandb restore username/project:run_id` を呼び出すと、新しいブランチをチェックアウトしてコードを復元し、元のコマンドがあらかじめ設定された状態で、トレーニングに使用したのと同一の docker イメージを起動します。