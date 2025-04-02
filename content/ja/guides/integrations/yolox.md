---
title: YOLOX
description: W&B と YOLOX を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) は、 オブジェクト検出において強力なパフォーマンスを発揮する、アンカーフリー版の YOLO です。 YOLOX W&B インテグレーションを使用すると、トレーニング、検証、およびシステムに関連する メトリクス の ログ 記録をオンにすることができ、単一の コマンドライン 引数で 予測 をインタラクティブに検証できます。

## サインアップして APIキー を作成する

APIキー は、W&B へのあなたの マシン を認証します。APIキー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. [**User Settings**]（ユーザー 設定 ）を選択し、[**API Keys**]（APIキー ）セクションまでスクロールします。
3. [**Reveal**]（表示）をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。



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

## メトリクス を ログ 記録する

`--logger wandb` コマンドライン 引数を使用して、wandb での ログ 記録をオンにします。オプションで、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) が予期するすべての 引数 を渡すこともできます。各 引数 の前に `wandb-` を付けます。

`num_eval_imges` は、 モデル の評価のために W&B の テーブル に ログ 記録される 検証セット の画像と 予測 の数を制御します。

```shell
# wandb にログイン
wandb login

# `wandb` ロガー 引数 を使用して yolox トレーニング スクリプトを呼び出す
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

[YOLOX のトレーニング および 検証 メトリクス を使用した ダッシュボード の例 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="" >}}

この W&B インテグレーション に関する質問や問題がありますか？[YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) で issue をオープンしてください。
