---
title: Hugging Face
menu:
  tutorials:
    identifier: huggingface
    parent: integration-tutorials
weight: 3
---

{{< img src="/images/tutorials/huggingface.png" alt="Hugging Face と W&B のインテグレーション" >}}

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb" >}}
[Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスを [W&B](https://wandb.ai/site) インテグレーションで素早く可視化しましょう。

複数モデルのハイパーパラメーター、アウトプットメトリクス、GPU 利用率などのシステム統計を比較できます。

## なぜ W&B を使うべき？
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

- **統合型ダッシュボード**：すべてのモデルメトリクスと予測を一元管理
- **軽量**：Hugging Face との連携にコード修正は不要
- **アクセスしやすい**：個人やアカデミックチームは無料で利用可能
- **セキュア**：すべてのプロジェクトはデフォルトで非公開
- **信頼性**：OpenAI、トヨタ、Lyft など多くの機械学習チームが導入

W&B は GitHub のように機械学習モデルを管理できます— あなたの実験を非公開かつホスティングされたダッシュボードに保存。スクリプトをどこで実行していても、すべてのバージョン管理が自動で行われるので安心して素早く実験可能です。

W&B との軽量なインテグレーションは、どんな Python スクリプトにも対応しており、無料の W&B アカウントにサインアップするだけですぐに実験管理や可視化が始められます。

Hugging Face Transformers リポジトリでは、Trainer から W&B にトレーニングと評価のメトリクスを自動でログするようにしています。

インテグレーションの詳細はこちら：[Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)。

## インストール・インポート・ログイン

このチュートリアルで使用する Hugging Face・W&B ライブラリ、GLUE データセットおよびトレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [W&B]({{< relref "/" >}}): 実験管理と可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解のベンチマークデータセット
- [GLUE script](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): シーケンス分類のためのモデルトレーニングスクリプト

```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```

```notebook
# run_glue.py スクリプトでは transformers の開発版が必要です
!pip install -q git+https://github.com/huggingface/transformers
```

続ける前に、[無料アカウントにサインアップ](https://app.wandb.ai/login?signup=true)してください。

## APIキー を入力

サインアップが完了したら、次のセルを実行してリンクをクリックし、APIキーを取得してノートブックを認証します。

```python
import wandb
wandb.login()
```

必要に応じて、環境変数を設定してW&Bのログ方法をカスタマイズできます。詳細は [Hugging Face インテグレーションガイド]({{< relref "/guides/integrations/huggingface/" >}})をご覧ください。

```python
# オプション：パラメータと勾配の両方をログ
%env WANDB_WATCH=all
```

## モデルをトレーニング

ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を実行し、トレーニングの様子が W&B ダッシュボードに自動でトラッキングされるのを確認しましょう。このスクリプトは、Microsoft Research Paraphrase Corpus（意味的に同等かを人間が注釈した文ペア）で BERT をファインチューンします。

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

上に表示されるリンク、もしくは [wandb.ai](https://app.wandb.ai) にアクセスすると、リアルタイムで結果が可視化されます。すべての依存関係が読み込まれた後に「**wandb**: View run at [URL to your unique run]」という出力が表示され、そのURLから run をブラウザで確認できます。

**モデルパフォーマンスの可視化**  
多数の実験を比較したり、興味深い学びにズームインしたり、高次元データを手軽に可視化できます。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="モデルメトリクス・ダッシュボード" >}}

**アーキテクチャ比較**  
例えば [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) の比較も簡単です。アーキテクチャの違いがトレーニング中の精度にどのような影響を及ぼすか、線グラフの自動可視化で一目瞭然です。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="BERT と DistilBERT の比較" >}}

## 重要な情報を自動でトラッキング

W&B では、各実験ごとに新しい run が保存されます。デフォルトで保存される情報は以下の通りです：
- **ハイパーパラメーター**：モデルの設定が Config に保存
- **モデルメトリクス**：メトリクスの時系列データが Log に保存
- **ターミナルログ**：コマンドライン出力もタブで確認可能
- **システムメトリクス**：GPU・CPU 利用率、メモリ、温度など

## さらに詳しく

- [Hugging Face インテグレーションガイド]({{< relref "/guides/integrations/huggingface" >}})
- [YouTube の動画ガイド](http://wandb.me/youtube)
