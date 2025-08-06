---
title: PaddleOCR
description: W&B を PaddleOCR と連携する方法
menu:
  default:
    identifier: paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語対応で優れた実用的な OCR ツールを目指し、ユーザーがより良いモデルをトレーニングし、実際に活用できるように PaddlePaddle 上で開発されています。PaddleOCR は OCR に関するさまざまな最先端アルゴリズムをサポートしており、産業用途のソリューションも提供しています。現在 PaddleOCR は W&B とのインテグレーションが提供されており、トレーニングや評価のメトリクス、モデルのチェックポイントとそれに関連するメタデータも一緒にログできます。

## 例: ブログ & Colab

[PaddleOCR を使い ICDAR2015 データセットでモデルをトレーニングする方法はこちらをご覧ください](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)。[Google Colab](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) も利用でき、対応する W&B ダッシュボードのライブデータも [こちら](https://wandb.ai/manan-goel/text_detection) で確認できます。また、このブログの中国語版は [こちら](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4) です。

## サインアップと APIキー を作成

APIキー は、あなたのマシンと W&B を認証するものです。ユーザープロフィールから APIキー を生成できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のプロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示された APIキー をコピーします。APIキー を非表示にするにはページを再読み込みしてください。

## `wandb` ライブラリをインストールし、ログインする

`wandb` ライブラリをローカルにインストールし、ログインする方法です。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) にあなたの APIキー を設定します。

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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## wandb を `config.yml` ファイルに追加

PaddleOCR では、設定用の変数を yaml ファイルで指定します。設定 yaml の末尾に下記のスニペットを追加すると、すべてのトレーニングと検証のメトリクスが W&B ダッシュボードに自動でログされ、モデルのチェックポイントも保存されます。

```python
Global:
    use_wandb: True
```

追加で [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) に渡したい引数があれば、yaml ファイル内の `wandb` 以下にオプションとして記載できます。

```
wandb:  
    project: CoolOCR  # （任意）この wandb プロジェクト名 
    entity: my_team   # （任意）チームで wandb を利用している場合はここにチーム名
    name: MyOCRModel  # （任意）この wandb run の名前
```

## `config.yml` を `train.py` に渡す

作成した yaml ファイルは、PaddleOCR リポジトリにある [トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py) の引数として渡します。

```bash
python tools/train.py -c config.yml
```

`train.py` を W&B を有効にして実行すると、自動的に W&B ダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="PaddleOCR training dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="PaddleOCR validation dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="Text Detection Model dashboard" >}}

## フィードバック・不具合について

W&B インテグレーションに関するご意見や不具合がある場合は、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) に issue を投稿するか、<a href="mailto:support@wandb.com">support@wandb.com</a> までメールでご連絡ください。