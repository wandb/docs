---
description: W&B を Docker と統合する方法
slug: /guides/integrations/docker
displayed_sidebar: default
---


# Docker

## Docker Integration

W&Bは、コードが実行されたDockerイメージへのポインターを保存できるため、以前の実験をそのままの環境で復元することができます。wandbライブラリは、この状態を永続化するために **WANDB\_DOCKER** 環境変数を探します。自動的にこの状態を設定するためのヘルパーをいくつか提供しています。

### ローカル開発

`wandb docker` は、dockerコンテナを起動し、wandb環境変数を渡し、コードをマウントし、wandbをインストールすることを保証するコマンドです。デフォルトでは、TensorFlow、PyTorch、Keras、およびJupyterをインストールしたdockerイメージを使用します。同じコマンドを使用して独自のdockerイメージを起動することもできます: `wandb docker my/image:latest`。このコマンドは現在のディレクトリーをコンテナの"/app"ディレクトリーにマウントしますが、"--dir"フラグを使用して変更することができます。

### プロダクション

`wandb docker-run` コマンドは、プロダクションワークロードのために提供されています。これは `nvidia-docker` の代替として利用することを意図しています。このコマンドは、`docker run` コマンドの簡単なラッパーであり、資格情報と **WANDB\_DOCKER** 環境変数を呼び出しに追加します。"--runtime"フラグを渡さずに `nvidia-docker` がマシンに利用可能な場合、ランタイムがnvidiaに設定されることも保証されます。

### Kubernetes

トレーニングワークロードをKubernetesで実行し、k8s APIがポッドに公開されている場合（デフォルトではそのようになっています）、wandbはAPIからdockerイメージのダイジェストをクエリし、自動的に **WANDB\_DOCKER** 環境変数を設定します。

## 復元

**WANDB\_DOCKER** 環境変数で計測されたrunの場合、`wandb restore username/project:run_id` を呼び出すことで、コードを復元し、元のコマンドがあらかじめ設定された状態でトレーニングに使用された正確なdockerイメージを起動する新しいブランチをチェックアウトします。