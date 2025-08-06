---
title: Docker
description: W&B を Docker と統合する方法
menu:
  default:
    identifier: docker
    parent: integrations
weight: 80
---

## Docker インテグレーション

W&B は、あなたのコードが実行された Docker イメージへのポインタを保存できます。これにより、過去の Experiment を正確に再現した環境で復元することが可能です。wandbライブラリは、この状態を保存するために **WANDB_DOCKER** 環境変数を探します。自動的にこの状態を設定するためのいくつかのヘルパーも用意しています。

### ローカル開発

`wandb docker` コマンドは、docker コンテナを起動し、wandb の環境変数を渡し、あなたのコードをマウントし、wandb がインストールされていることを保証します。デフォルトでは、このコマンドは TensorFlow、PyTorch、Keras、Jupyter がインストールされた docker イメージを使用します。同じコマンドで自分の docker イメージを起動することもできます: `wandb docker my/image:latest`。 コマンドは現在のディレクトリーをコンテナ内の "/app" ディレクトリーにマウントしますが、"--dir" フラグでこれを変更できます。

### プロダクション

`wandb docker-run` コマンドは、プロダクションワークロード用に用意されています。これは `nvidia-docker` の置き換えとして使用できるよう設計されています。このコマンドは、`docker run` コマンドのシンプルなラッパーで、あなたの認証情報と **WANDB_DOCKER** 環境変数を自動的に追加します。"--runtime" フラグを指定しない場合で、マシンに `nvidia-docker` がインストールされているときは、実行時に nvidia が設定されることも保証します。

### Kubernetes

トレーニングワークロードを Kubernetes 上で実行しており、k8s API があなたの pod に公開されている場合（これはデフォルトでそうなっています）、wandb は API から docker イメージのダイジェスト値を取得し、自動で **WANDB_DOCKER** 環境変数を設定します。

## 復元

run が **WANDB_DOCKER** 環境変数とともに記録されていた場合、`wandb restore username/project:run_id` を実行することで、新しいブランチでコードが復元され、トレーニング時に使用されたのとまったく同じ docker イメージが、元のコマンドとともに起動されます。