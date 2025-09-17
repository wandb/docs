---
title: Hugging Face
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-huggingface
    parent: integration-tutorials
weight: 3
---

{{< img src="/images/tutorials/huggingface.png" alt="Hugging Face と W&B のインテグレーション" >}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb" >}}
シームレスな [W&B](https://wandb.ai/site) インテグレーションで、あなたの [Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスをすばやく可視化しましょう。

ハイパーパラメーター、出力メトリクス、GPU 使用率のようなシステム統計を、複数のモデル間で比較できます。 

## Why should I use W&B?
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

- **統合ダッシュボード**: すべてのモデルのメトリクスと予測の中央リポジトリ
- **軽量**: Hugging Face とのインテグレーションに コードの変更は不要
- **利用しやすい**: 個人とアカデミックチームは無料
- **セキュア**: すべての Projects はデフォルトでプライベート
- **信頼されています**: OpenAI、Toyota、Lyft などの 機械学習 チームに採用

W&B は 機械学習 モデルのための GitHub のようなものだと考えてください — 機械学習の 実験 をあなたのプライベートなホスト型ダッシュボードに保存できます。どこで スクリプト を実行していても、モデルのあらゆる バージョン が保存される安心感のもと、素早く実験できます。

W&B の軽量なインテグレーションはあらゆる Python スクリプトで動作し、モデルの 実験管理 と可視化を始めるために必要なのは無料の W&B アカウントへのサインアップだけです。

Hugging Face Transformers のリポジトリでは、Trainer に計測を入れており、各ロギングステップで トレーニング と評価のメトリクスを自動的に W&B に ログ します。

インテグレーションの仕組みを詳しく解説したものがこちらです: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

## Install, import, and log in



このチュートリアル用に、Hugging Face と W&B のライブラリ、GLUE データセット、そして トレーニングスクリプト をインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語のモデルとデータセット
- [W&B]({{< relref path="/" lang="ja" >}}): 実験管理 と可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解のベンチマーク データセット
- [GLUE script](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): 系列分類のための トレーニングスクリプト


```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```


```notebook
# run_glue.py スクリプトは transformers の開発版が必要です
!pip install -q git+https://github.com/huggingface/transformers
```

続行する前に、[無料アカウントにサインアップ](https://app.wandb.ai/login?signup=true)してください。

## Put in your API key

サインアップが済んだら、次のセルを実行してリンクをクリックし、APIキー を取得してこのノートブックを認証してください。


```python
import wandb
wandb.login()
```

任意で、環境 変数 を設定して W&B のロギングをカスタマイズできます。詳しくは [Hugging Face インテグレーションガイド]({{< relref path="/guides/integrations/huggingface/" lang="ja" >}}) を参照してください。


```python
# 任意: 勾配 と パラメータ の両方をログする
%env WANDB_WATCH=all
```

## モデルをトレーニングする
次に、ダウンロードした トレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を実行すると、トレーニングが自動的に W&B ダッシュボードへ トラッキング される様子が確認できます。このスクリプトは Microsoft Research Paraphrase Corpus 上で BERT を ファインチューン します。これは、2 つの文が意味的に等価かどうかを人手アノテーションで示した文ペアのコーパスです。


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

##  ダッシュボードで結果を可視化する
上に表示されたリンクをクリックするか、[wandb.ai](https://app.wandb.ai) にアクセスすると、結果がリアルタイムにストリーミングされるのが見られます。依存関係の読み込みが完了すると、ブラウザーで run を表示するためのリンクが現れます。次の出力を探してください: 「**wandb**: View run at [URL to your unique run]」

**モデルのパフォーマンスを可視化**
数十もの 実験 を横断して見たり、興味深い 学び にズームインしたり、高次元 データ を可視化するのも簡単です。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="モデル メトリクス のダッシュボード" >}}

**アーキテクチャー を比較**
こちらは [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) を比較した例です。自動生成される折れ線プロットの 可視化 により、異なる アーキテクチャー がトレーニング全体を通して評価 精度 にどのような影響を与えるかが簡単にわかります。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="BERT と DistilBERT の比較" >}}

## 重要な情報を手間なくデフォルトでトラッキング
W&B は 実験 ごとに新しい run を保存します。デフォルトで保存される情報は次のとおりです:
- **ハイパーパラメーター**: モデルの 設定 は Config に保存
- **モデル メトリクス**: ストリーミングされる メトリクス の 時系列 データは Log に保存
- **ターミナルログ**: コマンドラインの出力が保存され、タブで参照可能
- **システム メトリクス**: GPU や CPU の使用率、メモリ、温度など

## さらに学ぶ
- [Hugging Face インテグレーションガイド]({{< relref path="/guides/integrations/huggingface" lang="ja" >}})
- [YouTube の動画ガイド](http://wandb.me/youtube)