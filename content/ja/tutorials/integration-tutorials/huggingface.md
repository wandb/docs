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
あなたの [Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスを、シームレスな [W&B](https://wandb.ai/site) インテグレーションで素早く可視化しましょう。

ハイパーパラメーター、出力メトリクス、およびモデルの間でのシステム統計情報（GPU 利用状況など）を比較できます。

## なぜ W&B を使うべきなのか？
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

- **統一されたダッシュボード**: すべてのモデルメトリクスと予測のための中央リポジトリ
- **軽量**: Hugging Face とのインテグレーションにコード変更は不要
- **アクセシブル**: 個人および学術チーム向けに無料
- **セキュア**: すべてのプロジェクトはデフォルトでプライベート
- **信頼性**: OpenAI、トヨタ、Lyft などの機械学習チームが使用

W&B を機械学習モデルの GitHub のように考えてください。機械学習の実験をプライベートでホストされたダッシュボードに保存します。どこでスクリプトを実行しているかに関わらず、すべてのモデルのバージョンが保存されるという安心感で素早く実験を行えます。

W&B の軽量インテグレーションは、どんな Python スクリプトとも動作し、モデルのトラッキングと可視化を開始するために必要なのは、無料の W&B アカウントへの登録だけです。

Hugging Face Transformers リポジトリでは、Trainer を用いてトレーニングおよび評価メトリクスを各ログステップで W&B に自動的にログするように設定しています。

こちらは統合がどのように機能するかの詳細な説明です: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

## インストール、インポート、ログイン

このチュートリアルのために、Hugging Face と Weights & Biases のライブラリ、および GLUE データセットとトレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [Weights & Biases]({{< relref path="/" lang="ja" >}}): 実験管理と可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解ベンチマークデータセット
- [GLUE script](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): 系列分類用のモデルトレーニングスクリプト

```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```

```notebook
# run_glue.py スクリプトには transformers dev が必要
!pip install -q git+https://github.com/huggingface/transformers
```

続ける前に、[無料アカウントに登録](https://app.wandb.ai/login?signup=true)してください。

## API キーを入力

登録したら、次のセルを実行してリンクをクリックし、API キーを取得してこのノートブックを認証します。

```python
import wandb
wandb.login()
```

必要に応じて、環境変数を設定して W&B のログをカスタマイズできます。詳細は [ドキュメント]({{< relref path="/guides/integrations/huggingface/" lang="ja" >}}) を参照してください。

```python
# オプション: 勾配とパラメータの両方をログ
%env WANDB_WATCH=all
```

## モデルをトレーニング

次に、ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を呼び出し、Weights & Biases ダッシュボードにトレーニングが自動的にトラッキングされるのを確認します。このスクリプトは、BERT を Microsoft Research Paraphrase Corpus でファインチューンします。このデータセットは、文章が意味的に等価であるかを示す人間の注釈を含む文ペアです。

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

## ダッシュボードで結果を可視化

上記のリンクをクリックするか、[wandb.ai](https://app.wandb.ai) にアクセスして、結果がリアルタイムでストリームされるのを確認します。ブラウザであなたの run を見るためのリンクは、すべての依存関係がロードされた後に表示されます。以下の出力を確認してください: "**wandb**: 🚀 [あなたのユニークな run の URL] で run を見る"

**モデルパフォーマンスを可視化**
数十の実験を一目で確認し、興味深い学びをズームインし、高度に次元の高いデータを可視化することが簡単です。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="" >}}

**アーキテクチャーを比較**
こちらは [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) の比較例です。異なるアーキテクチャーがトレーニング中の評価精度にどのように影響を与えるかを自動的な折れ線グラフの可視化により簡単に確認できます。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="" >}}

## デフォルトで重要な情報を簡単にトラッキング

Weights & Biases は、各実験の新しい run を保存します。デフォルトで保存される情報は次のとおりです:
- **ハイパーパラメーター**: モデルの設定が Config に保存されます
- **モデルメトリクス**: ストリーム中の時系列データメトリクスが Log に保存されます
- **ターミナルログ**: コマンドライン出力が保存され、タブで確認可能です
- **システムメトリクス**: GPU と CPU の利用率、メモリ、温度など

## もっと詳しく知りたい方へ

- [ドキュメント]({{< relref path="/guides/integrations/huggingface" lang="ja" >}}): Weights & Biases と Hugging Face のインテグレーションに関するドキュメント
- [ビデオ](http://wandb.me/youtube): YouTube チャンネルでのチュートリアル、実務者とのインタビューなど
- お問い合わせ: 質問は contact@wandb.com までメッセージをお送りください。