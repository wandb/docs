---
title: YOLOX
description: W&B を YOLOX と統合する方法
menu:
  default:
    identifier: yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、強力なオブジェクト検出性能を持つアンカー不要型の YOLO バージョンです。YOLOX の W&B インテグレーションを利用すると、トレーニングや検証、システムに関するメトリクスのログを有効化でき、コマンドライン引数ひとつでインタラクティブに予測結果の検証も行えます。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に認証するためのものです。ユーザープロフィールから APIキー を発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize)に直接アクセスして APIキー を発行できます。表示された APIキー をコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリック。表示された APIキー をコピーしてください。APIキー を非表示にするにはページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に APIキー を設定します。

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

コマンドラインで `--logger wandb` 引数を指定すると、wandb でのログ記録が有効になります。必要に応じて、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) で指定できるすべての引数も渡せます。各引数の前に `wandb-` を付けてください。

`num_eval_imges` は、検証セット画像と予測結果が W&B テーブルにモデルの評価用としてログされる数を制御します。

```shell
# wandb にログイン
wandb login

# `wandb` ロガー引数付きで yolox トレーニングスクリプトを実行
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

[YOLOX のトレーニング・検証メトリクスを可視化したダッシュボード例はこちら →](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="YOLOX training dashboard" >}}

この W&B インテグレーションについて質問や問題があれば、[YOLOX リポジトリ](https://github.com/Megvii-BaseDetection/YOLOX) で issue をオープンしてください。