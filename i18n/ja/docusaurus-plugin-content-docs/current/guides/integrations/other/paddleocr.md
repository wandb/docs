---
slug: /guides/integrations/paddleocr
description: How to integrate W&B with PaddleOCR.
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleOCR

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、PaddlePaddleで実装されたマルチ言語、優れた、先導的で実用的なOCRツールを作成することを目指しています。PaddleOCRは、OCRに関連するさまざまな最先端のアルゴリズムをサポートし、産業ソリューションを開発しました。現在、PaddleOCRはW&Bとの統合が可能で、トレーニングと評価のメトリクスとモデルのチェックポイントと対応するメタデータを記録できます。

## 例：ブログ & Colab

[**こちら**](https://wandb.ai/manan-goel/text\_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)で、ICDAR2015データセットでPaddleOCRを使用してモデルをトレーニングする方法を読むことができます。この内容は、[**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)と[**こちらの**](https://wandb.ai/manan-goel/text\_detection)ライブW&Bダッシュボードも対応しています。このブログの中国語版はこちら[**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb\_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)。

## Weights & BiasesとPaddleOCRを使用する方法

### 1. W&Bにサインアップしてログインする

[**こちら**](https://wandb.ai/site)で無料アカウントにサインアップし、Python 3 環境でwandbライブラリをインストールしてください。ログインするには、www.wandb.aiでアカウントにサインインしている必要があります。**APIキーは** [**Authorizeページ**](https://wandb.ai/authorize) **で見つけることができます。**

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### 2. `config.yml`ファイルにwandbを追加する

PaddleOCRは、yamlファイルを使用して設定変数を提供する必要があります。以下のスニペットを設定yamlファイルの最後に追加すると、トレーニングと検証のメトリクスとモデルチェックポイントがすべて自動的にW&Bダッシュボードに記録されます。

```python
Global:
    use_wandb: True
```

yamlファイルの`wandb`ヘッダーの下に、[`wandb.init`](https://docs.wandb.ai/guides/track/launch)に渡したい任意の追加引数を追加することもできます。

```
wandb:  
    project: CoolOCR  # (任意) これはwandbプロジェクト名です
    entity: my_team   # (任意) もしwandbチームを使用している場合は、ここにチーム名を入力してください
    name: MyOCRModel  # (任意) これはwandb runの名前です
```
### 3. `config.yml`ファイルを `train.py` に渡す

yamlファイルは、PaddleOCRリポジトリで利用可能な[トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)の引数として提供されます。

```
python tools/train.py -c config.yml
```

Weights & Biases を有効化した状態で `train.py` ファイルを実行すると、W&Bダッシュボードにアクセスするためのリンクが生成されます：

![](/images/integrations/paddleocr_wb_dashboard1.png) ![](/images/integrations/paddleocr_wb_dashboard2.png)

![テキスト検出モデルのW&Bダッシュボード](/images/integrations/paddleocr_wb_dashboard3.png)

## フィードバックや問題点は？

Weights & Biases のインテグレーションに関するフィードバックや問題があれば、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR) に issue を開いていただくか、support@wandb.com までメールでお問い合わせください。