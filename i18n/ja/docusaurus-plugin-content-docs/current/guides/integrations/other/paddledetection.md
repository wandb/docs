---
slug: /guides/integrations/paddledetection
description: How to integrate W&B with PaddleDetection.
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)は、[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)をベースにしたエンドツーエンドのオブジェクト検出開発キットです。ネットワークコンポーネント、データ拡張、損失などの設定可能なモジュールを用いて、主要なオブジェクト検出、インスタンスセグメンテーション、トラッキング、キーポイント検出アルゴリズムをモジュラー設計で実装しています。

現在のPaddleDetectionには、トレーニングと検証のメトリクス、およびモデルのチェックポイントとそれに対応するメタデータをすべて記録するW&B統合が標準で搭載されています。

## 例としてのブログとColab

[**こちらのブログをお読みください**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)では、PaddleDetectionを使用してCOCO2017データセットのサブセットでYOLOXモデルをトレーニングする方法を紹介しています。これには、[**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing)も含まれており、対応するW&Bダッシュボードは[**こちら**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=)でリアルタイムに閲覧できます。

## PaddleDetection WandbLogger

PaddleDetection WandbLoggerは、トレーニングと評価のメトリクスをWeights & Biasesに記録し、トレーニング中のモデルのチェックポイントも記録します。

## Weights & BiasesとPaddleDetectionを使用する

### W&Bにサインアップしてログインする

[**無料のWeights & Biasesアカウントにサインアップ**](https://wandb.ai/site)し、wandbライブラリをpipでインストールします。ログインするには、www.wandb.aiでアカウントにサインインする必要があります。サインインしたら、[**Authorizeページ**](https://wandb.ai/authorize) **でAPIキーが表示されます。**

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```
!pip install wandb

wandb.login()
```
  </TabItem>
</Tabs>

### トレーニングスクリプトでWandbLoggerを有効にする

#### CLIを使用する

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)の`train.py`に引数としてwandbを使用するには：

* `--use_wandb`フラグを追加します
* 最初のwandb引数の前に`-o`が必要（一度だけ指定）
* 個々のwandb引数には`wandb-`プレフィックスが必要です。例えば、[`wandb.init`](https://docs.wandb.ai/ref/python/init)に渡す 引数には`wandb-`プレフィックスをつけます

```
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```
#### config.ymlファイルを利用する方法

また、configファイルを通じてwandbを活性化することもできます。wandbヘッダーの下に、config.ymlファイルにwandbの引数を追加してください：

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Weights & Biasesをオンにした状態で`train.py`ファイルを実行すると、W&Bダッシュボードにアクセスするためのリンクが生成されます。

![Weights & Biasesダッシュボード](/images/integrations/paddledetection_wb_dashboard.png)

## フィードバックや問題について

Weights & Biasesのインテグレーションに関するご意見や問題がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)にIssueを立てるか、support@wandb.comまでメールしてください。