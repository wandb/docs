---
title: YOLOX
description: W&B と YOLOX の連携方法。
menu:
  default:
    identifier: ja-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、オブジェクト検出で高い性能を発揮する YOLO の anchor-free 版です。YOLOX の W&B インテグレーションを使うと、トレーニング、検証、システムに関するメトリクスのログ記録を有効化でき、単一の コマンドライン 引数で 予測 を対話的に検証できます。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に認証します。APIキー は ユーザープロファイル から生成できます。

{{% alert %}}
よりスムーズに進めるには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を隠すにはページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの APIキー を設定します。

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

{{% tab header="Python ノートブック" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## メトリクスをログに記録

`--logger wandb` コマンドライン 引数を使って、wandb でのログ記録を有効にします。必要に応じて、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) が受け付けるすべての 引数 も渡せます。その際は各 引数 の先頭に `wandb-` を付けてください。

`num_eval_imges` は、モデルの評価 のために W&B Tables にログされる 検証セット の画像と 予測 の数を制御します。

```shell
# wandb にログイン
wandb login

# `wandb` ロガー引数を付けて YOLOX のトレーニングスクリプトを実行
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

[YOLOX のトレーニングと検証メトリクスのダッシュボード例 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="YOLOX のトレーニング ダッシュボード" >}}

この W&B インテグレーションに関する質問や問題は、[YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) で Issue をオープンしてください。