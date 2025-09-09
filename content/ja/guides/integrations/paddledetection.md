---
title: PaddleDetection
description: W&B と PaddleDetection を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は、[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) に基づく エンドツーエンド の物体検出開発キットです。ネットワーク コンポーネント、データ拡張、損失などの構成可能なモジュールを用いて、各種の主流オブジェクトを検出し、インスタンスをセグメンテーションし、キーポイントの追跡と検出を行います。

PaddleDetection にはビルトインの W&B インテグレーションが含まれており、トレーニングおよび検証のメトリクスに加えて、モデルのチェックポイントと対応するメタデータをすべてログします。

PaddleDetection の `WandbLogger` は、トレーニング中に トレーニングおよび評価メトリクスを W&B に、さらに モデル の チェックポイント をログします。

[PaddleDetection で YOLOX モデルを `COCO2017` データセットのサブセットで統合する方法を紹介する W&B のブログ記事](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) をお読みください。

## サインアップして API キーを作成

API キーは、あなたのマシンを W&B に認証します。API キーはあなたの ユーザー プロフィールから生成できます。

{{% alert %}}
よりスムーズな方法として、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上の ユーザー プロフィール アイコンをクリックします。
1. **User Settings** を選択し、その後 **API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを隠すにはページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` の [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの API キーに設定します。

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

## トレーニング スクリプトで `WandbLogger` を有効化

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) の `train.py` に引数で wandb を渡して使うには:

* `--use_wandb` フラグを追加します
* 最初の wandb 引数の前に `-o` を付けます（これは 1 回だけでかまいません）
* それぞれの個別の引数は必ず `"wandb-"` プレフィックスを含めてください。たとえば [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡す任意の引数には `wandb-` プレフィックスを付けます

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
`config.yml` ファイルの `wandb` キーの下に wandb の引数を追加します:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` ファイルを実行すると、W&B ダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="W&B ダッシュボード" >}}
{{% /tab %}}
{{< /tabpane >}}

## フィードバックや問題

W&B インテグレーションに関するフィードバックや問題がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) に Issue を作成するか、<a href="mailto:support@wandb.com">support@wandb.com</a> までメールでお知らせください。