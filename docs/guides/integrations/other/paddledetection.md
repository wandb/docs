---
description: W&B を PaddleDetection と統合する方法
slug: /guides/integrations/paddledetection
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) に基づいたエンドツーエンドのオブジェクト検出開発キットです。ネットワークコンポーネント、データ拡張、損失などのモジュール設計で、多様な主流のオブジェクト検出、インスタンスセグメンテーション、トラッキングおよびキーポイント検出アルゴリズムを実装しています。

PaddleDetection は、トレーニングおよび検証メトリクス、モデルチェックポイント、それに対応するメタデータをログするためのW&Bインテグレーションが組み込まれています。

## 例: ブログとColab

PaddleDetectionを使ってCOCO2017データセットのサブセットでYOLOXモデルをトレーニングする方法については、[**こちらのブログ**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)をお読みください。また、[**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing)も利用可能で、対応するライブW&Bダッシュボードは[**こちら**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=)からアクセスできます。

## PaddleDetection WandbLogger

PaddleDetection WandbLoggerは、トレーニングと評価のメトリクス、およびトレーニング中のモデルチェックポイントを Weights & Biases にログします。

## Weights & BiasesとPaddleDetectionの利用

### W&Bにサインアップとログイン

[**こちらからサインアップ**](https://wandb.ai/site)して、無料のWeights & Biasesアカウントを作成し、wandbライブラリをインストールしてください。ログインするには、www.wandb.aiであなたのアカウントにサインインしている必要があります。サインインすると、[**認証ページ**](https://wandb.ai/authorize)で**APIキーが見つかります**。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```
  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

wandb.login()
```
  </TabItem>
</Tabs>

### トレーニングスクリプトでWandbLoggerを有効にする

#### CLIを使用する場合

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)の `train.py` に引数を渡してwandbを使用するには:

* `--use_wandb` フラグを追加する
* 最初のwandb引数は `-o` に続ける（これを一度だけ渡せばok）
* 個々のwandb引数には `wandb-` プレフィックスを付ける。例えば[`wandb.init`](https://docs.wandb.ai/ref/python/init) に渡す引数には `wandb-` プレフィックスを付けます

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```

#### config.ymlファイルを使用する場合

configファイルを介してwandbを有効にすることもできます。config.ymlファイルにwandb引数を下記のように追加します:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Weights & Biasesを有効にして `train.py` ファイルを実行すると、W&Bダッシュボードへのリンクが生成されます:

![A Weights & Biases Dashboard](/images/integrations/paddledetection_wb_dashboard.png)

## フィードバックや問題点

Weights & Biasesインテグレーションに関するフィードバックや問題がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)で問題を報告するか、support@wandb.com にメールしてください。