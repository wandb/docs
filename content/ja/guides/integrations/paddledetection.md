---
title: PaddleDetection
description: W&B を PaddleDetection と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は、[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) に基づくエンドツーエンドの物体検出開発キットです。ネットワークコンポーネント、データ拡張、損失などの設定可能なモジュールを使用して、さまざまな主流オブジェクトを検出し、インスタンスをセグメント化し、キーポイントを追跡および検出します。

PaddleDetection には、トレーニングと検証のメトリクス、モデルチェックポイント、およびそれに対応するメタデータをログするための W&B インテグレーションが組み込まれています。

PaddleDetection `WandbLogger` は、トレーニングと評価のメトリクスを Weights & Biases にログし、トレーニング中にモデルチェックポイントも記録します。

[**W&B ブログ記事を読む**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) では、YOLOX モデルを `COCO2017` データセットのサブセットで PaddleDetectionと統合する方法を示しています。

## サインアップして API キーを作成する

APIキーは、あなたのマシンをW&Bに認証します。APIキーはユーザープロフィールから生成できます。

{{% alert %}}
より簡略化された方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
2. **ユーザー設定** を選び、**API キー** セクションまでスクロールします。
3. **表示** をクリックし、表示された API キーをコピーします。API キーを非表示にするには、ページをリロードします。

## `wandb` ライブラリをインストールしログインする

`wandb` ライブラリをローカルにインストールしてログインする方法:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたのAPIキーに設定します。

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

{{% tab header="Pythonノートブック" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## トレーニングスクリプトで `WandbLogger` を有効にする

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) の `train.py` に引数を渡して wandb を使用するには:

* `--use_wandb` フラグを追加します
* 最初の wandb 引数の前に `-o` を付けます（これは一度だけで結構です）
* 個々の wandb 引数にはすべて `wandb-` プレフィックスを含める必要があります。例えば、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡す引数には `wandb-` プレフィックスが追加されます

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
`config.yml` ファイルの `wandb` キーの下に wandb 引数を追加します:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` ファイルを実行すると、W&B ダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A Weights & Biases Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## フィードバックや問題

Weights & Biases インテグレーションに関するフィードバックや問題がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) に issue を作成するか、<a href="mailto:support@wandb.com">support@wandb.com</a> にメールしてください。