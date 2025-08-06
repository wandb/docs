---
title: PaddleDetection
description: W&B を PaddleDetection と連携する方法
menu:
  default:
    identifier: paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) をベースにした、エンドツーエンドの物体検出開発キットです。各種主流の物体を検出し、インスタンスのセグメンテーションやキーポイントの検出・追跡などが可能です。ネットワークコンポーネントやデータ拡張、ロスなどのモジュールを設定して使用できます。

PaddleDetection には W&B とのインテグレーションが標準搭載されており、トレーニングやバリデーションのメトリクス、モデルのチェックポイント、そのメタデータも自動でログされます。

PaddleDetection の `WandbLogger` を使うと、トレーニングや評価時のメトリクス、さらにトレーニング中のモデルチェックポイントがすべて W&B に記録されます。

[W&B のブログ記事を読む](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)  
この記事では、YOLOX モデルを PaddleDetection と組み合わせて `COCO2017` データセットの一部サブセット上で使う例を紹介しています。

## サインアップして APIキー を作成

APIキーはあなたのマシンを W&B に認証するためのものです。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) へ直接アクセスして APIキー を生成することも可能です。表示された APIキー をコピーして、パスワードマネージャー等の安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリック。表示された APIキー をコピーしてください。APIキー を隠すにはページをリロードしてください。

## `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールしてログインする方法です：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に取得した APIキー を設定します。

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

## トレーニングスクリプトで `WandbLogger` を有効化

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) の `train.py` に引数を渡して wandb を利用する場合：

* `--use_wandb` フラグを付けます
* 最初の wandb 引数の前に `-o` を付けます（1回だけでOKです）
* 各 wandb 用引数には `"wandb-"` プレフィックスを付与します。たとえば [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) 用の引数には `wandb-` を頭に付けてください

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```
{{% /tab %}}
{{% tab header="`config.yml`" value="config" %}}
wandb の各種引数を config.yml ファイル内の `wandb` キー以下に追加してください:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` を実行すると、W&B ダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A W&B Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## フィードバックや不具合について

W&B インテグレーションに関するご意見や不具合は、[PaddleDetection の GitHub](https://github.com/PaddlePaddle/PaddleDetection) に Issue を投稿するか、<a href="mailto:support@wandb.com">support@wandb.com</a> までご連絡ください。