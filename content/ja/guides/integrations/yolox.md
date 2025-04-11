---
title: YOLOX
description: W&B を YOLOX と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、オブジェクト検出において優れたパフォーマンスを持つ、アンカーフリー版のYOLOです。YOLOX W&Bインテグレーションを使用すると、トレーニング、検証、およびシステムに関連するメトリクスのログをオンにすることができ、単一のコマンドライン引数で予測をインタラクティブに検証することができます。

## サインアップしてAPIキーを作成する

APIキーは、W&Bに対してマシンを認証します。APIキーはユーザープロファイルから生成できます。

{{% alert %}}
よりスムーズなアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスしてAPIキーを生成することができます。表示されたAPIキーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインする方法:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [environment variable]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。

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

## メトリクスをログする

`--logger wandb` コマンドライン引数を使用して、wandbでのロギングを有効にします。また、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) が期待するすべての引数を渡すこともできます。それぞれの引数には `wandb-` を前置します。

`num_eval_imges` は、モデルの評価のためにW&Bテーブルにログされる検証セット画像と予測の数を制御します。

```shell
# wandb にログイン
wandb login

# `wandb` ロガー引数を使って yolox のトレーニングスクリプトを呼び出します
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

[YOLOX のトレーニングと検証メトリクスを含むダッシュボードの例 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="" >}}

このW&Bインテグレーションに関する質問や問題がありますか？ [YOLOXリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX)でissueを開いてください。