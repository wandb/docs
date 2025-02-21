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

W&B は、コードが実行された Docker イメージへのポインタを保存することができ、以前の実験を実行された正確な環境に復元する能力を提供します。 wandbライブラリは、この状態を永続化するために **WANDB_DOCKER** 環境変数を探します。 自動的にこの状態を設定するヘルパーをいくつか提供しています。

### ローカル開発

`wandb docker` は、dockerコンテナを起動し、wandb環境変数を渡し、コードをマウントし、wandbがインストールされていることを確認するコマンドです。デフォルトでは、TensorFlow、PyTorch、Keras、そして Jupyter がインストールされた docker イメージを使用します。 同じコマンドを使用して独自の docker イメージを起動することもできます: `wandb docker my/image:latest`。 コマンドは現在のディレクトリーをコンテナの "/app" ディレクトリーにマウントしますが、"--dir" フラグでこれを変更できます。

### プロダクション

`wandb docker-run` コマンドはプロダクションの作業負荷用に提供されています。 これは、`nvidia-docker` の代わりとして使用することを目的としています。 これは、`docker run` コマンドへの単純なラッパーであり、資格情報と **WANDB_DOCKER** 環境変数をコールに追加します。 "--runtime" フラグを渡さず、`nvidia-docker` がマシン上で利用可能な場合、ランタイムが nvidia に設定されることも保証されます。

### Kubernetes

トレーニングの作業負荷を Kubernetes 上で実行しており、k8s API がポッドに公開されている場合（デフォルトではそのようになっています）、wandb は Docker イメージのダイジェストを API に問い合わせ、**WANDB_DOCKER** 環境変数を自動的に設定します。

## 復元

もし run が **WANDB_DOCKER** 環境変数で操作されていた場合、`wandb restore username/project:run_id` を呼び出すことで、新しいブランチをチェックアウトし、コードを復元した後、トレーニングに使用された正確な docker イメージをオリジナルのコマンドで起動します。