---
description: W&B を PaddleOCR に統合する方法
slug: /guides/integrations/paddleocr
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# PaddleOCR

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) は、多言語対応で素晴らしく、最先端で実用的なOCRツールを作成し、ユーザーがより良いモデルをトレーニングし、それを実際に活用することを目指しています。PaddleOCRはOCRに関連するさまざまな最先端のアルゴリズムをサポートし、産業ソリューションを開発しています。現在、PaddleOCRにはWeights & Biasesのインテグレーションが追加されており、トレーニングおよび評価メトリクスをログし、対応するメタデータと一緒にモデルチェックポイントを記録できます。

## 例: ブログ & Colab

PaddleOCRを使用してICDAR2015データセットでモデルをトレーニングする方法については、こちらの[**ブログ**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)をご覧ください。また、[**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)も用意されており、対応するライブW&Bダッシュボードは[**こちら**](https://wandb.ai/manan-goel/text_detection)で利用できます。このブログの中国語バージョンもこちらにあります: [**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## Weights & BiasesとPaddleOCRの使用

### 1. サインアップとログイン

無料アカウントに[**サインアップ**](https://wandb.ai/site)し、Python 3 環境にwandbライブラリをインストールします。ログインするには、www.wandb.ai でアカウントにサインインする必要があります。次に、[**認証ページ**](https://wandb.ai/authorize)でAPIキーを見つけられます。

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

### 2. `config.yml`ファイルにwandbを追加

PaddleOCRでは、設定変数をyamlファイルで提供する必要があります。設定yamlファイルの末尾に次のスニペットを追加すると、モデルチェックポイントと一緒にすべてのトレーニングおよびバリデーションメトリクスが自動的にW&Bダッシュボードにログされます:

```python
Global:
    use_wandb: True
```

[`wandb.init`](https://docs.wandb.ai/guides/track/launch)に渡したいその他の任意の引数は、yamlファイルの`wandb`ヘッダーの下に追加できます:

```
wandb:  
    project: CoolOCR  # (オプション) これはwandbプロジェクト名です
    entity: my_team   # (オプション) wandbチームを使用している場合は、ここにチーム名を渡すことができます
    name: MyOCRModel  # (オプション) これはwandb runの名前です
```

### 3. `config.yml`ファイルを`train.py`に渡す

yamlファイルは、PaddleOCRリポジトリにある[トレーニングスクリプト](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)に引数として提供されます。

```
python tools/train.py -c config.yml
```

Weights & Biasesをオンにして`train.py`ファイルを実行すると、W&Bダッシュボードへのリンクが生成されます:

![](/images/integrations/paddleocr_wb_dashboard1.png) ![](/images/integrations/paddleocr_wb_dashboard2.png)

![W&Bダッシュボード テキスト検出モデル](/images/integrations/paddleocr_wb_dashboard3.png)

## フィードバックや問題がありますか?

Weights & Biasesのインテグレーションに関するフィードバックや問題がありましたら、[PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)でイシューを作成するか、support@wandb.comまでメールでご連絡ください。