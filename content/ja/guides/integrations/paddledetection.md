---
title: PaddleDetection
description: PaddleDetection と W&B の統合方法。
menu:
  default:
    identifier: ja-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) は、[PaddlePaddle](https://github.com/PaddlePaddle/PaddlePaddle) に基づくエンドツーエンドの オブジェクト検出開発キットです。ネットワークコンポーネント、データ拡張、損失などの構成可能なモジュールを使用して、さまざまな主流のオブジェクトを検出し、インスタンスをセグメント化し、キーポイントを追跡および検出します。

PaddleDetection には、すべての トレーニング および 検証 メトリクスに加えて、モデル チェックポイント とそれに対応する メタデータ をログに記録する、組み込みの W&B インテグレーションが含まれています。

PaddleDetection `WandbLogger` は、トレーニング 中にトレーニング および 評価 メトリクス を Weights & Biases に ログ 記録するだけでなく、モデル チェックポイント も ログ 記録します。

[**W&B の ブログ 記事を読む**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)。この記事では、`COCO2017` データセット の サブセット で、YOLOX モデル を PaddleDetection と 統合 する方法について説明します。

## サインアップ して APIキー を作成する

APIキー は、W&B に対して マシン を 認証 します。APIキー は、 ユーザー プロフィールから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にある ユーザー プロフィール アイコンをクリックします。
2. **ユーザー 設定** を選択し、**APIキー** セクションまでスクロールします。
3. **表示** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードしてください。

## `wandb` ライブラリ を インストール して ログイン する

`wandb` ライブラリ をローカルに インストール して ログイン するには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリ を インストール して ログイン します。

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

{{% tab header="Python ノートブック" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## トレーニング スクリプト で `WandbLogger` をアクティブにする

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) の `train.py` への 引数 を介して wandb を使用するには:

* `--use_wandb` フラグを追加します
* 最初の wandb 引数 は `-o` を前に付ける必要があります (これは一度だけ渡す必要があります)
* 個々の wandb 引数 には、プレフィックス `wandb-` が含まれている必要があります。たとえば、[`wandb.init`]({{< relref path="/ref/python/init" lang="ja" >}}) に渡される 引数 には、`wandb-` プレフィックス が付きます。

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```
{{% /tab %}}
{{% tab header="`config.yml`" value="config" %}}
`wandb` キー の下の config.yml ファイル に wandb 引数 を追加します。

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` ファイル を実行すると、W&B ダッシュボード への リンク が生成されます。

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A Weights & Biases Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## フィードバック または 問題点

Weights & Biases インテグレーション に関する フィードバック や 問題 がある場合は、[PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection) で 問題 を提起するか、<a href="mailto:support@wandb.com">support@wandb.com</a> にメールを送信してください。
