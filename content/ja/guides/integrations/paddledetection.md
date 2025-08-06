---
title: PaddleDetection
description: W&B を PaddleDetection と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) をベースにしたエンドツーエンドの物体検出開発キットです。ネットワークコンポーネント、データ拡張、損失関数などの設定可能なモジュールを使って、様々な主流オブジェクトの検出やインスタンスセグメンテーション、キーポイント検出、追跡が可能です。

PaddleDetection には、トレーニングや検証時の全てのメトリクスやモデルのチェックポイント、それらのメタデータをログする W&B 公式インテグレーションが組み込まれています。

PaddleDetection の `WandbLogger` は、トレーニングおよび評価のメトリクスを W&B に、トレーニング中のモデルチェックポイントもあわせてログします。

[W&B のブログ記事](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) では、YOLOX モデルを PaddleDetection で `COCO2017` データセットのサブセットに統合する方法が紹介されています。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B へ認証するためのものです。ユーザープロフィールから APIキー を生成できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして APIキー を生成することもできます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキー をコピーします。APIキー を隠したい場合はページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールし、ログインします。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に自身の APIキー を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールして、ログインします。



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

## トレーニングスクリプトで `WandbLogger` を有効化

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) の `train.py` へ引数を渡して wandb を利用する場合：

* `--use_wandb` フラグを追加します
* 最初の wandb 引数の前に `-o` を付けます（1回のみ指定すればOKです）
* 各 wandb 用引数には `"wandb-"` の接頭辞が必要です。例えば [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡したい任意の引数も `wandb-` で始めてください

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
`config.yml` ファイルの `wandb` キーの下に wandb 用引数を追加します。

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` を実行すると、W&B ダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A W&B ダッシュボード" >}}
{{% /tab %}}
{{< /tabpane >}}

## フィードバック・不具合の報告

W&B インテグレーションに関するフィードバックや不具合があれば、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) で issue を立てるか、<a href="mailto:support@wandb.com">support@wandb.com</a> までメールでご連絡ください。