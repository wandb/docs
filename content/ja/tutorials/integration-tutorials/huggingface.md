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
[Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスを [W&B](https://wandb.ai/site) とシームレスに連携して、すぐに可視化しましょう。

ハイパーパラメーター、出力メトリクス、システム統計（GPU使用率など）を、複数のモデル間で比較できます。

## なぜ W&B を使うのか？
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

- **統合ダッシュボード**：すべてのモデルメトリクスや予測を集約管理
- **軽量**：Hugging Face とのインテグレーションにコードの変更は不要
- **アクセスしやすい**：個人やアカデミックチームは無料
- **セキュア**：すべてのプロジェクトはデフォルトで非公開
- **信頼性**：OpenAI、トヨタ、Lyft などの機械学習チームも利用

W&B は機械学習モデルのための GitHub のような存在です— 機械学習の実験をプライベートでホストされたダッシュボードに保存。スクリプトをどこで実行しても、すべてのモデルのバージョンが自動的に保存されるので、安心して素早く実験できます。

W&B の軽量インテグレーションは、どんな Python スクリプトとも動作します。始めるのに必要なのは、W&B の無料アカウント登録だけ。すぐにモデルのトラッキングと可視化が可能です。

Hugging Face Transformers のリポジトリでは、Trainer が各ログステップごとに W&B へトレーニングと評価のメトリクスを自動的に記録します。

統合の詳細な仕組みはこちらで解説しています: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)。

## インストール・インポート・ログイン



このチュートリアルでは、Hugging Face と W&B のライブラリ、GLUE データセット、トレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers)：自然言語モデルとデータセット
- [W&B]({{< relref path="/" lang="ja" >}})：実験管理と可視化
- [GLUE データセット](https://gluebenchmark.com/)：言語理解ベンチマークデータセット
- [GLUE スクリプト](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py)：時系列分類のためのモデルトレーニングスクリプト


```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```


```notebook
# run_glue.py スクリプトのため、transformersの開発版が必要です
!pip install -q git+https://github.com/huggingface/transformers
```

先に、[無料アカウントに登録](https://app.wandb.ai/login?signup=true)しましょう。

## APIキーを入力

アカウント登録後、次のセルを実行してリンクをクリックし、APIキーを取得し、このノートブックを認証してください。


```python
import wandb
wandb.login()
```

必要に応じて、環境変数を設定することで W&B のログ記録をカスタマイズできます。詳細は [Hugging Face インテグレーションガイド]({{< relref path="/guides/integrations/huggingface/" lang="ja" >}}) をご覧ください。


```python
# オプション: 勾配とパラメータの両方をログします
%env WANDB_WATCH=all
```

## モデルをトレーニング

次に、ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を実行すれば、トレーニングの様子が自動的に W&B ダッシュボードへ記録されます。このスクリプトは BERT を Microsoft Research Paraphrase Corpus（意味的に等価かどうかの人間によるアノテーション済みな文ペア）でファインチューニングします。


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

##  ダッシュボードで結果を可視化

上記で出力されたリンク、または [wandb.ai](https://app.wandb.ai) にアクセスして、リアルタイムで結果を確認できます。依存関係のロード後、runのリンクがブラウザに表示されます。 "**wandb**: View run at [URL to your unique run]" という出力を探してください。

**モデルパフォーマンスの可視化**
多数の Experiments の中から興味深い学びを拡大表示したり、高次元データを簡単に可視化できます。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="モデルメトリクスダッシュボード" >}}

**アーキテクチャーの比較**
こちらは [BERT と DistilBERT の比較](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) の例です。異なるアーキテクチャーで、トレーニング中の評価精度がどのように変化するかを、自動生成される折れ線グラフで簡単に比較できます。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="BERT vs DistilBERT の比較" >}}

## 重要な情報をデフォルトで簡単にトラッキング

W&B は Experiments ごとに新しい run を保存します。デフォルトで下記の情報が保存されます:
- **ハイパーパラメーター**：モデルの設定値が Config に記録されます
- **モデルメトリクス**：メトリクスの時系列データが Log にストリーミングされます
- **ターミナルログ**：コマンドライン出力がタブで保存され閲覧可能
- **システムメトリクス**：GPU・CPUの使用率、メモリ、温度など

## 詳しく知る

- [Hugging Face インテグレーションガイド]({{< relref path="/guides/integrations/huggingface" lang="ja" >}})
- [YouTube の動画ガイド](http://wandb.me/youtube)