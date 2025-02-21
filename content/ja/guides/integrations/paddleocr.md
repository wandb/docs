---
title: PaddleOCR
description: W&B と PaddleOCR の統合方法。
menu:
  default:
    identifier: ja-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語に対応した、素晴らしく、最先端で、実用的な OCR ツールを作成し、ユーザー がより優れたモデル をトレーニング し、PaddlePaddle で実装された 実際 のアプリケーション に適用できるよう支援することを目指しています。PaddleOCR は、OCR に関連するさまざまな最先端のアルゴリズムをサポートし、産業用ソリューションを開発しました。PaddleOCR には、対応する メタデータ とともにモデル のチェックポイント を使用して、トレーニング および 評価 メトリクス をログ に記録するための Weights & Biases インテグレーション が付属しています。

## ブログ と Colab の例

ICDAR2015 データセット で PaddleOCR を使用してモデル をトレーニング する方法については、[**こちら**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) を参照してください。これには、[**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) が付属しており、対応するライブ W&B ダッシュボード は[**こちら**](https://wandb.ai/manan-goel/text_detection)で入手できます。このブログ の中国語版はこちら:[**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## サインアップ して APIキー を作成する

APIキー は、W&B に対してマシン を認証します。APIキー は、ユーザー プロファイル から生成できます。

{{% alert %}}
より効率的なアプローチ については、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピー し、パスワード マネージャー などの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にある ユーザー プロファイル アイコン をクリックします。
2. [**User Settings**]を選択し、[**API Keys**]セクション までスクロールします。
3. [**Reveal**]をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページ をリロードします。

## `wandb` ライブラリ をインストール して ログイン する

`wandb` ライブラリ をローカル にインストール して ログイン するには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. APIキー に `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリ をインストール して ログイン します。

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

## `config.yml` ファイル に wandb を追加する

PaddleOCR では、yaml ファイル を使用して 設定変数 を指定する必要があります。設定 yaml ファイル の末尾に次のスニペット を追加すると、すべてのトレーニング および 検証 メトリクス がモデル チェックポイント とともに W&B ダッシュボード に自動的に記録されます。

```python
Global:
    use_wandb: True
```

[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡したい追加 のオプション の 引数 は、yaml ファイル の `wandb` ヘッダー の下に追加することもできます。

```
wandb:  
    project: CoolOCR  # (オプション) これは wandb プロジェクト 名です
    entity: my_team   # (オプション) wandb team を使用している場合、ここに team 名を渡すことができます
    name: MyOCRModel  # (オプション) これは wandb run の名前です
```

## `config.yml` ファイル を `train.py` に渡す

yaml ファイル は、PaddleOCR リポジトリ で入手できる[トレーニング スクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)への 引数 として提供されます。

```bash
python tools/train.py -c config.yml
```

Weights & Biases をオン にして `train.py` ファイル を実行すると、W&B ダッシュボード に移動するためのリンク が生成されます。

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="Text Detection Model の W&B ダッシュボード" >}}

## フィードバック または 問題点

Weights & Biases インテグレーション に関する フィードバック または 問題 がある場合 は、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) で 問題 をオープン するか、<a href="mailto:support@wandb.com">support@wandb.com</a> にメール してください。
