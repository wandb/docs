---
title: Hugging Face
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-huggingface
    parent: integration-tutorials
weight: 3
---

{{< img src="/images/tutorials/huggingface.png" alt="" >}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb" >}}
[Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスを、シームレスな [W&B](https://wandb.ai/site) の インテグレーションで素早く可視化しましょう。

モデル間で、ハイパーパラメーター 、出力 メトリクス、および GPU 使用率のようなシステム統計を比較します。

## W&B を使うべき理由
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

- **統一された ダッシュボード**: すべてのモデル メトリクスと 予測 のための一元的なリポジトリ
- **軽量**: Hugging Face と インテグレーションするために必要なコード変更はありません
- **アクセス可能**: 個人およびアカデミック な Teams は無料
- **セキュア**: デフォルトでは、すべての Projects はプライベートです
- **信頼**: OpenAI、Toyota、Lyft などの 機械学習 の Teams で使用されています

W&B は 機械学習 モデル用の GitHub のようなものだと考えてください。機械学習 の 実験 をプライベートなホストされた ダッシュボード に保存します。スクリプトを実行している場所に関係なく、モデルのすべての バージョン が保存されているという自信を持って、迅速に 実験 できます。

W&B の軽量な インテグレーション は、あらゆる Python スクリプト で動作し、モデルのトラッキングと 可視化 を開始するために必要なのは、無料の W&B アカウントにサインアップすることだけです。

Hugging Face Transformers リポジトリでは、各ログ ステップ で トレーニング および 評価 メトリクスを自動的に W&B に記録するように Trainer をインストルメントしています。

インテグレーション がどのように機能するかの詳細はこちら: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)。

## インストール、インポート、ログイン

Hugging Face と Weights & Biases ライブラリ、およびこの チュートリアル のための GLUE データセット と トレーニング スクリプト をインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルと データセット
- [Weights & Biases]({{< relref path="/" lang="ja" >}}): 実験管理 と 可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解 ベンチマーク データセット
- [GLUE script](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): シーケンス分類のためのモデル トレーニング スクリプト

```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```

```notebook
# the run_glue.py script requires transformers dev
!pip install -q git+https://github.com/huggingface/transformers
```

続行する前に、[無料アカウントにサインアップ](https://app.wandb.ai/login?signup=true) してください。

## APIキー を入力してください

サインアップしたら、次のセルを実行し、リンクをクリックして APIキー を取得し、この ノートブック を認証します。

```python
import wandb
wandb.login()
```

オプションで、環境変数 を設定して W&B のログをカスタマイズできます。詳細は [ドキュメント]({{< relref path="/guides/integrations/huggingface/" lang="ja" >}}) を参照してください。

```python
# Optional: log both gradients and parameters
# オプション: 勾配 と パラメータ の両方をログに記録します
%env WANDB_WATCH=all
```

## モデル の トレーニング
次に、ダウンロードした トレーニング スクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を呼び出して、 トレーニング が自動的に Weights & Biases ダッシュボード に追跡されるのを確認します。このスクリプトは、Microsoft Research Paraphrase Corpus (意味的に同等かどうかを示す人間の注釈が付いた文のペア) で BERT を ファインチューン します。

```python
%env WANDB_PROJECT=huggingface-demo
%env TASK_NAME=MRPC

!python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --logging_steps 50
```

## ダッシュボード で 結果 を 可視化する
上記に出力されたリンクをクリックするか、[wandb.ai](https://app.wandb.ai) にアクセスして、結果 がライブでストリームされるのを確認します。ブラウザで run を確認するためのリンクは、すべての依存関係がロードされた後に表示されます。次の出力を探してください: "**wandb**: 🚀 View run at [URL to your unique run]"

**モデル パフォーマンス の 可視化**
数十の 実験 を簡単に見渡し、興味深い 学び にズームインし、高次元の データ を 可視化 できます。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="" >}}

**アーキテクチャ の比較**
[BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) を比較した例を次に示します。自動折れ線グラフ 可視化 により、さまざまな アーキテクチャー が トレーニング 全体を通して 評価 精度にどのように影響するかを簡単に確認できます。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="" >}}

## デフォルト で、主要な 情報 を簡単に追跡する
Weights & Biases は、 実験 ごとに新しい run を保存します。デフォルト で保存される 情報 は次のとおりです。
- **ハイパーパラメーター**: モデル の 設定 は Config に保存されます
- **モデル メトリクス**: ストリーミングされる メトリクス の 時系列 データ は Log に保存されます
- **ターミナル ログ**: コマンドライン の 出力 は保存され、タブで利用できます
- **システム メトリクス**: GPU と CPU の 使用率、メモリ、温度 など。

## より詳しく知る
- [ドキュメント]({{< relref path="/guides/integrations/huggingface" lang="ja" >}}): Weights & Biases と Hugging Face の インテグレーション に関するドキュメント
- [ビデオ](http://wandb.me/youtube): チュートリアル 、 実践者 への インタビュー 、および YouTube チャンネル の詳細
- 連絡先: ご質問は contact@wandb.com までメッセージをお送りください
