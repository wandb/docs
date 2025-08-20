---
title: Docker
description: W&B を Docker と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker インテグレーション

W&B は、あなたのコードが実行された Docker イメージへのポインタを保存できるため、過去の実験を実行当時の正確な環境で再現できます。wandbライブラリは、この状態を永続化するために **WANDB_DOCKER** 環境変数を参照します。私たちは、この状態を自動的に設定するいくつかのヘルパーも用意しています。

### ローカル開発

`wandb docker` は、dockerコンテナを起動し、wandb の環境変数を渡し、あなたのコードをマウントし、wandbがインストールされていることを保証するコマンドです。デフォルトでは、TensorFlow、PyTorch、Keras、Jupyter がインストールされたdockerイメージが使われます。同じコマンドで自分の dockerイメージを起動することもできます：`wandb docker my/image:latest`。このコマンドは、現在のディレクトリーをコンテナの「/app」ディレクトリーにマウントします。`--dir` フラグでこのパスを変更できます。

### プロダクション

`wandb docker-run` コマンドは、プロダクション用途向けに提供されています。これは `nvidia-docker` の代替として使用できるコマンドです。`docker run` コマンドのラッパーであり、あなたの認証情報と **WANDB_DOCKER** 環境変数をコールに追加します。`--runtime` フラグを渡さずにマシン上に `nvidia-docker` がある場合、自動的にランタイムを nvidia に設定します。

### Kubernetes

Kubernetes でトレーニングワークロードを実行し、k8s API が pod から利用可能（デフォルトの設定です）な場合、wandb は API から dockerイメージのダイジェストを取得し、自動的に **WANDB_DOCKER** 環境変数を設定します。

## 復元

もし Run で **WANDB_DOCKER** 環境変数が設定されていれば、`wandb restore username/project:run_id` を呼び出すことで、新しいブランチをチェックアウトしてコードを復元し、トレーニング時に使用された dockerイメージを、元のコマンドを事前設定した状態で起動できます。