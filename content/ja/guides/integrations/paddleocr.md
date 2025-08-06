---
title: PaddleOCR
description: W&B を PaddleOCR と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語対応の高品質で実用的な OCR ツールを目指し、ユーザーがより優れたモデルをトレーニングし、それらを実際の用途で活用できるようにするため、PaddlePaddle で実装されています。PaddleOCR は OCR 関連のさまざまな最先端のアルゴリズムをサポートしており、産業向けのソリューションも開発しています。PaddleOCR には W&B のインテグレーションが組み込まれており、トレーニングや評価メトリクス、モデルチェックポイントと対応するメタデータをログすることができます。

## 参考ブログ & Colab

[PaddleOCR を使って ICDAR2015 データセットでモデルをトレーニングする方法はこちら](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) をご覧ください。同じ内容を Google Colab でもご利用いただけます: [Google Colab](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)。対応するライブ W&B ダッシュボードは [こちら](https://wandb.ai/manan-goel/text_detection) で確認できます。また、このブログの中国語版もあります: [W&B对您的OCR模型进行训练和调试](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## サインアップと APIキー の作成

APIキー は自分のマシンと W&B を認証するために使用されます。APIキー は自身のユーザープロフィールから発行できます。

{{% alert %}}
より簡単に取得したい場合は、[W&B 認証ページ](https://wandb.ai/authorize)に直接アクセスして APIキー を発行できます。表示された APIキー をコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして APIキー を表示し、コピーしてください。APIキー を非表示にするにはページを再読み込みしてください。

## `wandb` ライブラリをインストールしてログイン

ローカルで `wandb` ライブラリをインストールし、ログインする手順です。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を、ご自身の APIキー で設定します。

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

PaddleOCR では yaml ファイル形式で設定変数を指定する必要があります。以下のスニペットを設定 yaml ファイルの末尾に追加することで、全てのトレーニング・バリデーションメトリクスとモデルチェックポイントが、自動的に W&B ダッシュボードにログされます:

```python
Global:
    use_wandb: True
```

また、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡す追加のオプション引数がある場合は、yaml ファイル内の `wandb` ヘッダーの下に指定できます:

```
wandb:  
    project: CoolOCR  # (オプション) wandb の Project 名
    entity: my_team   # (オプション) wandb の Team を使う場合はチーム名を指定
    name: MyOCRModel  # (オプション) wandb run の名前
```

## `config.yml` ファイルを `train.py` に渡す

この yaml ファイルを、PaddleOCR リポジトリの [training script](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py) へ引数として渡します。

```bash
python tools/train.py -c config.yml
```

`train.py` を W&B を有効にして実行すると、W&B ダッシュボードへアクセスするためのリンクが生成されます:

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="PaddleOCR training dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="PaddleOCR validation dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="Text Detection Model dashboard" >}}

## フィードバックやご質問

W&B とのインテグレーションについてフィードバックや問題がありましたら、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)で issue を立てるか、<a href="mailto:support@wandb.com">support@wandb.com</a> までご連絡ください。