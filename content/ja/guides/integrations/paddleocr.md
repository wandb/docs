---
title: PaddleOCR
description: W&B を PaddleOCR と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) の目的は、多言語対応の素晴らしい、先進的で実用的なOCRツールを作成し、ユーザーがより良いモデルをトレーニングし、PaddlePaddleで実践的に適用できるようにすることです。PaddleOCRはOCRに関連する様々な最先端のアルゴリズムをサポートしており、産業向けのソリューションも開発しています。現在、PaddleOCRはWeights & Biasesと統合されており、トレーニングと評価メトリクスのログ取得、モデルのチェックポイントと対応するメタデータの管理を簡単に行えます。

## ブログの例 & Colab

[**こちらをお読みください**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) では、ICDAR2015データセットを使用してPaddleOCRでモデルをトレーニングする方法を説明しています。また、[**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)も付属しており、対応するライブW&Bダッシュボードは[**こちら**](https://wandb.ai/manan-goel/text_detection)で利用可能です。このブログの中国語版もこちらにあります：[**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## サインアップとAPIキーの作成

APIキーは、あなたのマシンをW&Bに認証します。APIキーはユーザープロフィールから生成できます。

{{% alert %}}
より簡素化された方法として、直接[こちら](https://wandb.ai/authorize)にアクセスしてAPIキーを生成できます。表示されたAPIキーをコピーして、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザーアイコンをクリックします。
2. **ユーザー設定**を選択し、**APIキー**のセクションまでスクロールします。
3. **表示**をクリックします。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みしてください。

## `wandb`ライブラリのインストールとログイン

ローカルに`wandb`ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をAPIキーに設定します。

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

## `config.yml`ファイルにwandbを追加

PaddleOCRは、設定変数をyamlファイルで提供することを要求します。設定yamlファイルの末尾に以下のスニペットを追加すると、すべてのトレーニングと検証メトリクスをW&Bダッシュボードに自動的にログすることができます。モデルのチェックポイントも同様です。

```python
Global:
    use_wandb: True
```

追加のオプション引数は、yamlファイルの`wandb`ヘッダーの下に追加できます。

```
wandb:  
    project: CoolOCR  # (オプション) これはwandbプロジェクト名です
    entity: my_team   # (オプション) wandbチームを使用している場合、チーム名をここに渡すことができます
    name: MyOCRModel  # (オプション) これはwandb runの名前です
```

## `config.yml`ファイルを`train.py`に渡す

yamlファイルは、PaddleOCRのリポジトリで利用可能な[トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)に引数として提供されます。

```bash
python tools/train.py -c config.yml
```

Weights & Biasesを有効にして`train.py`ファイルを実行すると、W&Bダッシュボードへのリンクが生成されます。

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="W&B Dashboard for the Text Detection Model" >}}

## フィードバックや問題

Weights & Biasesのインテグレーションに関するフィードバックや問題がある場合は、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)で問題を開くか、<a href="mailto:support@wandb.com">support@wandb.com</a>にメールでお問い合わせください。