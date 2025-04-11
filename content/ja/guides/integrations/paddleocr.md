---
title: PaddleOCR
description: PaddleOCR と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語対応で素晴らしく、実用的でユーザーがより良いモデルをトレーニングし、PaddlePaddleで実践的に適用できるOCRツールの作成を目指しています。PaddleOCRはOCRに関連するさまざまな最先端のアルゴリズムをサポートし、産業用ソリューションを開発しました。PaddleOCRにはWeights & Biasesのインテグレーションがあり、トレーニングと評価メトリクスをログに記録し、対応するメタデータとともにモデルのチェックポイントを保存できます。

## 例: ブログ & Colab

PaddleOCRでICDAR2015データセットを使ってモデルをトレーニングする方法を知るには、[**こちらをお読みください**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)。さらに[**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)も提供されており、対応するライブW&Bダッシュボードは[**こちら**](https://wandb.ai/manan-goel/text_detection)で利用できます。このブログの中国語バージョンもこちらで利用できます: [**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## サインアップしてAPIキーを作成する

APIキーは、W&Bへの認証に使われます。APIキーはユーザーのプロファイルから生成できます。

{{% alert %}}
よりスムーズな方法として、直接[https://wandb.ai/authorize](https://wandb.ai/authorize)にアクセスしてAPIキーを生成することができます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
2. **User Settings**を選択し、**API Keys**セクションまでスクロールします。
3. **Reveal**をクリックします。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みします。

## `wandb`ライブラリをインストールしてログインする

`wandb`ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を自分のAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールしてログインします。

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

## `config.yml`ファイルにwandbを追加する

PaddleOCRでは設定変数をyamlファイルで提供する必要があります。設定yamlファイルの最後に次のスニペットを追加することで、すべてのトレーニングおよびバリデーションメトリクスをW&Bダッシュボードに自動的にログ記録するとともに、モデルのチェックポイントも保存されます:

```python
Global:
    use_wandb: True
```

[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}})に渡したい追加の任意の引数は、yamlファイルの`wandb`ヘッダーの下に追加することもできます:

```
wandb:  
    project: CoolOCR  # (optional) これはwandbプロジェクト名です
    entity: my_team   # (optional) wandbチームを使っている場合、ここでチーム名を渡すことができます
    name: MyOCRModel  # (optional) これはwandb runの名前です
```

## `config.yml`ファイルを`train.py`に渡す

yamlファイルは、PaddleOCRリポジトリ内で利用可能な[トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)への引数として提供されます。

```bash
python tools/train.py -c config.yml
```

Weights & Biasesをオンにして`train.py`ファイルを実行するとき、W&Bダッシュボードへのリンクが生成されます:

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="W&B Dashboard for the Text Detection Model" >}}

## フィードバックや問題点

Weights & Biasesのインテグレーションに関するフィードバックや問題がある場合は、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)で問題を報告するか、<a href="mailto:support@wandb.com">support@wandb.com</a>にメールしてください。