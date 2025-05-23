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
[Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスをシームレスな [W&B](https://wandb.ai/site) インテグレーションで素早く可視化しましょう。

ハイパーパラメーター、アウトプットメトリクス、GPU利用率などのシステム統計をモデル間で比較します。

## なぜW&Bを使うべきか？
{.skipvale}

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

- **統一されたダッシュボード**: モデルのすべてのメトリクスと予測のための中央リポジトリ
- **軽量**: Hugging Faceとのインテグレーションにコード変更は不要
- **アクセス可能**: 個人や学術チームには無料
- **セキュア**: すべてのプロジェクトはデフォルトでプライベート
- **信頼性**: OpenAI、トヨタ、Lyftなどの機械学習チームで使用されている

W&Bを機械学習モデル用のGitHubのように考えてください。プライベートでホストされたダッシュボードに機械学習の実験管理を保存します。スクリプトをどこで実行しても、モデルのすべてのバージョンが保存されることを確信して、素早く実験できます。

W&Bの軽量なインテグレーションは、任意のPythonスクリプトで動作し、モデルのトラッキングと可視化を開始するには無料のW&Bアカウントにサインアップするだけです。

Hugging Face Transformersレポジトリでは、Trainingと評価メトリクスを各ログステップでW&Bに自動的にログするようにTrainerを設定しました。

インテグレーションの仕組みを詳しく見るにはこちら: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)

## インストール、インポート、ログイン

このチュートリアルのためにHugging FaceとWeights & Biasesのライブラリ、GLUEデータセット、トレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [Weights & Biases]({{< relref path="/" lang="ja" >}}): 実験管理と可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解ベンチマークデータセット
- [GLUE script](https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py): シーケンス分類用モデルのトレーニングスクリプト


```notebook
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/examples/pytorch/text-classification/run_glue.py
```


```notebook
# run_glue.pyスクリプトはtransformers devを必要とします
!pip install -q git+https://github.com/huggingface/transformers
```

続行する前に、[無料アカウントにサインアップしてください](https://app.wandb.ai/login?signup=true)。

## APIキーを入力

サインアップしたら、次のセルを実行してリンクをクリックし、APIキーを取得してこのノートブックを認証してください。


```python
import wandb
wandb.login()
```

オプションで、W&Bロギングをカスタマイズするために環境変数を設定できます。[ドキュメント]({{< relref path="/guides/integrations/huggingface/" lang="ja" >}})を参照してください。


```python
# オプション: 勾配とパラメータの両方をログします
%env WANDB_WATCH=all
```

## モデルをトレーニング
次に、ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を呼び出し、トレーニングがWeights & Biasesダッシュボードに自動的にトラックされるのを確認します。このスクリプトは、Microsoft Research Paraphrase CorpusでBERTをファインチューンし、意味的に同等であることを示す人間の注釈付きの文のペアを使用します。


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
上記で印刷されたリンクをクリックするか、[wandb.ai](https://app.wandb.ai) にアクセスして、結果がリアルタイムでストリームされるのを確認してください。ブラウザでrunを表示するリンクは、すべての依存関係がロードされた後に表示されます。次のような出力を探します: "**wandb**: 🚀 View run at [URL to your unique run]"

**モデルのパフォーマンスを可視化**
数十の実験管理を一目で確認し、興味深い学びにズームインし、高次元のデータを可視化するのは簡単です。

{{< img src="/images/tutorials/huggingface-visualize.gif" alt="" >}}

**アーキテクチャーを比較**
こちらは [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) を比較する例です。異なるアーキテクチャーがトレーニング中の評価精度にどのように影響するかを、自動ラインプロット可視化で簡単に確認できます。

{{< img src="/images/tutorials/huggingface-comparearchitectures.gif" alt="" >}}

## 重要な情報をデフォルトで簡単にトラック
Weights & Biasesは、各実験で新しいrunを保存します。デフォルトで保存される情報は次の通りです:
- **ハイパーパラメーター**: モデルの設定がConfigに保存されます
- **モデルメトリクス**: ストリーミングメトリクスの時系列データはLogに保存されます
- **ターミナルログ**: コマンドラインの出力は保存され、タブで利用可能です
- **システムメトリクス**: GPUとCPUの使用率、メモリ、温度など

## 詳しく知る
- [ドキュメント]({{< relref path="/guides/integrations/huggingface" lang="ja" >}}): Weights & BiasesとHugging Faceのインテグレーションに関するドキュメント
- [ビデオ](http://wandb.me/youtube): YouTubeチャンネルでのチュートリアル、実務者とのインタビュー、その他
- お問い合わせ: contact@wandb.com までご質問をお寄せください
