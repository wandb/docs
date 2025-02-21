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

W&B は、コードが実行された Docker イメージへのポインターを保存できます。これにより、以前の実験を、それが実行された正確な 環境 に復元することができます。 wandb ライブラリ は、この状態を永続化するために **WANDB_DOCKER** 環境 変数を探します。この状態を自動的に設定するいくつかのヘルパーを提供しています。

### ローカル開発

`wandb docker` は、 dockerコンテナ を起動し、 wandb の 環境 変数を渡し、 コード を マウント し、 wandb がインストールされていることを確認する コマンド です。デフォルトでは、この コマンド は TensorFlow 、 PyTorch 、 Keras 、 Jupyter がインストールされた docker イメージを使用します。同じ コマンド を使用して、独自の docker イメージを起動することもできます： `wandb docker my/image:latest` 。この コマンド は、現在の ディレクトリー を コンテナ の "/app" ディレクトリー に マウント します。これは "--dir" フラグで変更できます。

### 本番環境

`wandb docker-run` コマンド は、 本番環境 の ワークロード 用に提供されています。これは `nvidia-docker` の代替として使用されることを意図しています。これは `docker run` コマンド へのシンプルなラッパーで、 ユーザー の 認証情報と **WANDB_DOCKER** 環境 変数を呼び出しに追加します。 "--runtime" フラグを渡さず、 `nvidia-docker` がマシンで使用可能な場合、これも ランタイム が nvidia に設定されていることを確認します。

### Kubernetes

Kubernetes で トレーニング ワークロード を実行し、k8s API が pod に公開されている場合（デフォルトの場合）。 wandb は API に対して docker イメージのダイジェストをクエリし、 **WANDB_DOCKER** 環境 変数を自動的に設定します。

## 復元

run が **WANDB_DOCKER** 環境 変数で計測されている場合、 `wandb restore username/project:run_id` を呼び出すと、 コード を復元する新しいブランチをチェックアウトし、 トレーニング に使用された正確な docker イメージを、元の コマンド が事前に入力された状態で起動します。
