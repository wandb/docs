---
slug: /guides/integrations/docker
description: How to integrate W&B with Docker.
displayed_sidebar: default
---

# Docker
## Docker 連携

W&B は、コードが実行された Docker イメージへのポインタを保存できます。これにより、以前の実験を正確な環境で復元することが可能になります。wandb ライブラリは、この状態を永続化するために **WANDB\_DOCKER** 環境変数を探します。この状態を自動的に設定するいくつかのヘルパーを提供しています。

### ローカル開発

`wandb docker` は、dockerコンテナを起動し、wandb 環境変数を渡し、コードをマウントし、wandbがインストールされていることを保証するコマンドです。デフォルトでは、このコマンドは TensorFlow、PyTorch、Keras、および Jupyter がインストールされている Docker イメージを使用します。同じコマンドを使って、自分で作成した docker イメージを起動することができます: `wandb docker my/image:latest`。このコマンドは、現在のディレクトリをコンテナの "/app" ディレクトリにマウントします。"--dir" フラグを使用してこれを変更することができます。
### プロダクション

`wandb docker-run` コマンドは、プロダクションのワークロード用に提供されています。これは、`nvidia-docker`と置き換えるためのものです。これは、`docker run`コマンドへのシンプルなラッパーで、あなたの認証情報と**WANDB\_DOCKER**環境変数を追加します。"--runtime"フラグを渡さずに`nvidia-docker`がマシン上で利用可能な場合、これによりランタイムがnvidiaに設定されることを保証します。

### Kubernetes

Kubernetesでトレーニングワークロードを実行し、k8s APIがポッドに公開されている場合（デフォルトでそうなっている）、wandbはAPIからdockerイメージのダイジェストをクエリし、**WANDB\_DOCKER**環境変数を自動的に設定します。
## 復元

ランが**WANDB\_DOCKER**環境変数で計装されていた場合、`wandb restore username/project:run_id`を呼び出すと、新しいブランチが復元されてコードがチェックアウトされ、元のコマンドが事前に入力された正確なdockerイメージがトレーニング用に起動されます。