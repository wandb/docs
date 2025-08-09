---
title: YOLOX
description: W&B を YOLOX と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、高いオブジェクト検出性能を持つアンカーフリー版 YOLO です。YOLOX の W&B インテグレーションを使うことで、トレーニング、検証、システムに関連するメトリクスの ログ を有効化でき、コマンドライン引数 1 つで 予測 のインタラクティブな検証も行えます。

## サインアップして API キー を作成

APIキー は、あなたのマシンを W&B に認証するためのものです。APIキー は ユーザー プロフィールから作成できます。

{{% alert %}}
よりスムーズな方法として、[W&B 認証ページ](https://wandb.ai/authorize) へ直接アクセスし API キー を生成することも可能です。表示された APIキー をコピーし、パスワードマネージャなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キー をコピーします。API キー を隠すにはページを再読込してください。

## `wandb` ライブラリをインストールしてログイン

`wandb` ライブラリをローカルにインストールしてログインする手順：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの API キー を設定します。

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

## メトリクスをログする

コマンドライン引数 `--logger wandb` を使って wandb での ロギング を有効化できます。オプションで、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) で指定できるすべての引数も渡すことができます。その際は各引数の前に `wandb-` を付けてください。

`num_eval_imges` は、モデルの評価 のために W&B テーブル に ログ される検証セット画像および 予測 の数を制御します。

```shell
# wandb に ログイン
wandb login

# `wandb` ロガー引数を付けて yolox トレーニングスクリプトを実行
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

[YOLOX トレーニング & 検証メトリクス を含むダッシュボード例はこちら →](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="YOLOX training dashboard" >}}

この W&B インテグレーション についてご質問や問題があれば、[YOLOX のリポジトリ](https://github.com/Megvii-BaseDetection/YOLOX) に Issue を立ててください。