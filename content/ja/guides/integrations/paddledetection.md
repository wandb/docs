---
title: PaddleDetection
description: PaddleDetection と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は、[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) に基づくエンドツーエンドのオブジェクト検出開発キットです。ネットワークコンポーネント、データ拡張、損失などの設定可能なモジュールを使用して、さまざまな主流のオブジェクトを検出し、インスタンスをセグメント化し、キーポイントを追跡および検出します。

PaddleDetection では、トレーニングと検証のメトリクス、モデルのチェックポイントおよびそれに対応するメタデータをすべてログに記録するW&Bインテグレーションが内蔵されています。

PaddleDetection の `WandbLogger` は、トレーニングと評価のメトリクスを Weights & Biases と、トレーニング中のモデルのチェックポイントにログします。

[**W&Bのブログ記事を読む**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) こちらは、`COCO2017` データセットのサブセットで YOLOX モデルを PaddleDetection に統合する方法について説明しています。

## サインアップとAPIキーの作成

APIキーは、W&Bにあなたのマシンを認証します。APIキーはユーザープロファイルから生成できます。

{{% alert %}}
より簡略化されたアプローチとして、直接[https://wandb.ai/authorize](https://wandb.ai/authorize) に行って APIキーを生成することができます。表示されるAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロファイルアイコンをクリック。
1. **User Settings** を選択し、**API Keys** セクションまでスクロール。
1. **Reveal** をクリック。表示されたAPIキーをコピー。APIキーを非表示にするには、ページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

ローカルに `wandb` ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をAPIキーに設定します。

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

## トレーニングスクリプトで `WandbLogger` をアクティブ化

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}
PaddleDetection の `train.py` への引数として wandb を使用するには:

* `--use_wandb` フラグを追加
* 最初の wandb 引数は `-o` の後に続く必要があります（これは一度だけ渡せば良い）
* 各個別の wandb 引数には `wandb-` プレフィックスを含める必要があります。例えば、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡す引数は `wandb-` プレフィックスを付与します

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
`wandb` 引数を `wandb` キーの下で config.yml ファイルに追加します:

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

## フィードバックと問題

Weights & Biases インテグレーションについてフィードバックや問題がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) で問題をオープンするか、<a href="mailto:support@wandb.com">support@wandb.com</a> にメールを送ってください。