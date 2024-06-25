
# Hugging Face

<img src="https://i.imgur.com/vnejHGh.png" width="800"/>

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb)

シームレスな [W&B](https://wandb.ai/site) インテグレーションで [Hugging Face](https://github.com/huggingface/transformers) モデルのパフォーマンスを迅速に可視化。

モデル間でハイパーパラメーター、出力メトリクス、GPU利用率などのシステム統計を比較できます。



## 🤔 なぜW&Bを使うべきですか？

<img src="https://wandb.me/mini-diagram" width="650"/>

- **統一されたダッシュボード**: すべてのモデルのメトリクスや予測の中央リポジトリ
- **軽量**: Hugging Faceと統合するためのコード変更不要
- **アクセス可能**: 個人や学術チーム向けに無料
- **セキュア**: すべてのプロジェクトはデフォルトで非公開
- **信頼性**: OpenAI、Toyota、Lyftなどの機械学習チームに利用されています

W&Bを機械学習モデルのためのGitHubのように考えてください— 機械学習実験をプライベートでホスティングされたダッシュボードに保存します。スクリプトを実行する場所に関係なく、モデルのすべてのバージョンが保存されるという自信を持って迅速に実験できます。

W&Bの軽量インテグレーションは任意のPythonスクリプトと連携し、W&Bの無料アカウントにサインアップするだけで、モデルのトラッキングと可視化を始めることができます。

Hugging Face Transformersリポジトリでは、Trainerが各ログステップでトレーニングと評価のメトリクスを自動的にW&Bにログするように設定されています。

インテグレーションの詳細はこちらをご覧ください: [Hugging Face + W&B レポート](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

# 🚀 インストール、インポート、ログイン



このチュートリアルのためにHugging FaceとWeights & Biasesのライブラリ、GLUEデータセット、トレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [Weights & Biases](https://docs.wandb.com/): 実験管理と可視化
- [GLUEデータセット](https://gluebenchmark.com/): 言語理解のベンチマークデータセット
- [GLUEスクリプト](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): シーケンス分類のためのモデルトレーニングスクリプト


```python
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
```


```python
# run_glue.py スクリプトにはtransformers devが必要です
!pip install -q git+https://github.com/huggingface/transformers
```

## 🖊️ [無料アカウントにサインアップ →](https://app.wandb.ai/login?signup=true)

## 🔑 APIキーを入力
サインアップが完了したら、次のセルを実行してリンクをクリックし、APIキーを取得してこのノートブックを認証します。


```python
import wandb
wandb.login()
```

必要に応じて、環境変数を設定してW&Bのログをカスタマイズできます。詳細は[ドキュメント](https://docs.wandb.com/library/integrations/huggingface)を参照してください。


```python
# オプション: 勾配とパラメータの両方をログ
%env WANDB_WATCH=all
```

# 👟 モデルをトレーニング
次に、ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を呼び出し、トレーニングが自動的にWeights & Biasesダッシュボードにトラックされる様子を見ます。このスクリプトはMicrosoft Research Paraphrase Corpus — 人間の注釈で意味的に同等と示された文のペアにBERTをファインチューンします。


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

# 👀 ダッシュボードで結果を可視化
上に表示されたリンクをクリックするか、[wandb.ai](https://app.wandb.ai) にアクセスして結果をライブで確認できます。すべての依存関係がロードされた後にブラウザでrunを確認するリンクが表示されます — 次のような出力を探してください: "**wandb**: 🚀 View run at [URL to your unique run]"

**モデルパフォーマンスを可視化**
多数の実験を簡単に比較し、興味深い発見にズームインし、高次元データを可視化することができます。

![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

**アーキテクチャーを比較**
こちらは [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) を比較した例です — トレーニング中の評価精度に対するアーキテクチャーの影響を自動的なラインプロットの可視化で簡単に確認できます。
![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)



### 📈 デフォルトで主要情報を簡単にトラッキング
Weights & Biasesは各実験に新しいrunを保存します。デフォルトで保存される情報はこちらです:
- **ハイパーパラメーター**: Configにモデルの設定が保存されます
- **モデルメトリクス**: メトリクスの時系列データがLogに保存されます
- **ターミナルログ**: コマンドラインの出力が保存され、タブで閲覧可能
- **システムメトリクス**: GPUとCPUの利用率、メモリ、温度など


## 🤓 もっと知りたい！
- [ドキュメント](https://docs.wandb.com/huggingface): Weights & BiasesとHugging Faceのインテグレーションに関するドキュメント
- [ビデオ](http://wandb.me/youtube): チュートリアル、実務家とのインタビューなどがYouTubeチャンネルで提供されています
- お問い合わせ: contact@wandb.com までご質問ください