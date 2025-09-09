---
title: PaddleOCR
description: W&B を PaddleOCR と連携する方法。
menu:
  default:
    identifier: ja-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語で実用的かつ先進的な OCR ツールを提供し、ユーザーがより良い モデル をトレーニングして実運用に適用できるようにすることを目指しています。PaddleOCR は OCR 関連の 最先端の アルゴリズムを多数サポートし、産業向けソリューションも提供しています。現在 PaddleOCR には、トレーニング と 評価メトリクス を ログ し、対応する メタデータ 付きで モデル チェックポイント を記録できる W&B インテグレーションが含まれています。

## サンプル ブログ と Colab

[PaddleOCR を使って ICDAR2015 データセットで モデル をトレーニングする方法はこちら](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)。あわせて [Google Colab](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) も用意されており、対応するライブの W&B ダッシュボード は[こちら](https://wandb.ai/manan-goel/text_detection)です。また、この ブログ の中国語版もあります: [W&B对您的OCR模型进行训练和调试](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に対して認証するものです。APIキー は ユーザー プロフィール から生成できます。

{{% alert %}}
より簡単な方法として、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワード マネージャー などの安全な場所に保存してください。
{{% /alert %}}

1. 右上の ユーザー プロフィール アイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするにはページを再読み込みします。

## `wandb` ライブラリ をインストールしてログイン

ローカルに `wandb` ライブラリ をインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` の[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})にあなたの APIキー を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリ をインストールしてログインします。



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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## `config.yml` ファイル に wandb を追加

PaddleOCR では、設定 値は yaml ファイルで指定します。設定用 yaml ファイルの末尾に次のスニペットを追加すると、すべての トレーニング と 検証 の メトリクス が、対応する モデル チェックポイント とともに W&B ダッシュボード に自動で ログ されます:

```python
Global:
    use_wandb: True
```

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に渡したい 追加の 任意の 引数 は、yaml ファイル内の `wandb` セクションの下にも記述できます:

```
wandb:  
    project: CoolOCR  # (任意) これは W&B Project の 名前です 
    entity: my_team   # (任意) W&B Team を使っている場合は、ここに Team 名を指定します
    name: MyOCRModel  # (任意) これは W&B Run の 名前です
```

## `config.yml` ファイル を `train.py` に渡す

yaml ファイルは、PaddleOCR リポジトリにある[トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)に 引数 として渡します。

```bash
python tools/train.py -c config.yml
```

W&B を有効にして `train.py` を実行すると、W&B ダッシュボード へのリンクが生成されます:

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="PaddleOCR トレーニング ダッシュボード" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="PaddleOCR 検証 ダッシュボード" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="テキスト検出 モデル ダッシュボード" >}}

## フィードバックや問題

W&B インテグレーションに関するフィードバックや問題があれば、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) で issue を作成するか、<a href="mailto:support@wandb.com">support@wandb.com</a> までメールでご連絡ください。