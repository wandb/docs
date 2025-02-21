---
title: YOLOX
description: YOLOX と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、オブジェクト検出のためのパフォーマンスが優れたアンカーフリーのYOLOバージョンです。YOLOX W&Bインテグレーションを使用すると、トレーニング、検証、およびシステムに関連するメトリクスのログを有効にし、単一のコマンドライン引数で予測を対話的に検証することができます。

## サインアップとAPIキーの作成

APIキーは、お使いのマシンをW&Bに認証します。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
より効率的な方法として、直接[https://wandb.ai/authorize](https://wandb.ai/authorize)にアクセスしてAPIキーを生成することができます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページをリロードします。

## `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## メトリクスのログ

`--logger wandb` コマンドライン引数を使用して、wandbでのログを有効にします。オプションで、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) が期待するすべての引数も渡すことができます。各引数には `wandb-` を前置してください。

`num_eval_imges` は、検証セットの画像と予測の数を制御し、W&Bテーブルにモデルの評価用としてログします。

```shell
# wandbにログイン
wandb login

# `wandb` ロガー引数を使用してyoloxトレーニングスクリプトを呼び出します
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity>
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```

## 例

[YOLOX トレーニングと検証メトリクスの例ダッシュボード ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="" >}}

このW&Bインテグレーションについての質問や問題がありますか？[YOLOXリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)で問題をオープンしてください。