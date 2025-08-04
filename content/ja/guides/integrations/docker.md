---
title: Docker
description: W&B を Docker と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker インテグレーション

W&B は、コードが実行された Docker イメージへのポインターを保存することで、以前の実験を正確に実行された環境に復元することができます。wandbライブラリは、この状態を永続化するために **WANDB_DOCKER** 環境変数を探します。私たちは、この状態を自動的に設定するいくつかのヘルパーを提供しています。

### ローカル開発

`wandb docker` は、dockerコンテナを起動し、wandbの環境変数を渡し、コードをマウントし、wandb がインストールされていることを確認するコマンドです。デフォルトでは、TensorFlow、PyTorch、Keras、そして Jupyter がインストールされた docker イメージを使用します。`wandb docker my/image:latest` のようにして、同じコマンドで独自の docker イメージを開始することもできます。コマンドは現在のディレクトリーをコンテナの "/app" ディレクトリーにマウントしますが、これは "--dir" フラグで変更できます。

### プロダクション

`wandb docker-run` コマンドは、プロダクションのワークロードに提供されます。これは `nvidia-docker` の代替として使用されることを想定しています。これは、`docker run` コマンドにあなたの資格情報と **WANDB_DOCKER** 環境変数を追加する単純なラッパーです。"--runtime" フラグを渡さず、`nvidia-docker` がマシンにインストールされている場合、ランタイムが nvidia に設定されていることも確認されます。

### Kubernetes

トレーニングワークロードを Kubernetes 上で実行し、k8s API がポッドに公開されている場合（デフォルトでそうです）、wandb は API に対して docker イメージのダイジェストを問い合わせ、**WANDB_DOCKER** 環境変数を自動的に設定します。

## 復元

**WANDB_DOCKER** 環境変数を使用して run が計測されている場合、`wandb restore username/project:run_id` を呼び出すと、新しいブランチがチェックアウトされ、コードが復元され、トレーニングに使用された正確な docker イメージが、元のコマンドで事前に設定された状態で起動されます。