


# Hugging Face

<img src="https://i.imgur.com/vnejHGh.png" width="800"/>

[**こちらのColabノートブックで試してみてください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb)

シームレスな[W&B](https://wandb.ai/site)統合により、[Hugging Face](https://github.com/huggingface/transformers)モデルのパフォーマンスを素早く可視化できます。

ハイパーパラメーター、出力メトリクス、GPU利用率などのシステム統計をモデル間で比較できます。



## 🤔 なぜW&Bを使うべきか？

<img src="https://wandb.me/mini-diagram" width="650"/>

- **統合ダッシュボード**: モデルメトリクスと予測の中央リポジトリ
- **軽量**: Hugging Faceと統合するためのコード変更不要
- **アクセス可能**: 個人やアカデミーチームは無料で利用可能
- **安全**: すべてのプロジェクトはデフォルトでプライベート
- **信頼性**: OpenAI、トヨタ、Lyftなどの機械学習チームに利用されています

W&Bを機械学習モデル用のGitHubのようなものと考えてください。機械学習実験をプライベートなホストされたダッシュボードに保存できます。スクリプトをどこで実行していても、モデルのバージョンすべてが保存されているという安心感を持って迅速に実験できます。

W&Bの軽量統合は任意のPythonスクリプトで動作し、無料のW&Bアカウントにサインアップするだけで、モデルのトラッキングと可視化が開始できます。

Hugging Face Transformersレポには、Trainerが各ログステップでトレーニングと評価メトリクスを自動的にW&Bにログするように設定されています。

統合の動作についての詳細はこちらをご覧ください: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).

# 🚀 インストール、インポート、ログイン



Hugging FaceとWeights & Biasesライブラリ、そしてこのチュートリアルに必要なGLUEデータセットとトレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [Weights & Biases](https://docs.wandb.com/): 実験トラッキングと可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解のベンチマークデータセット
- [GLUE script](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): シーケンス分類用のモデルトレーニングスクリプト


```python
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
```


```python
# run_glue.pyスクリプトはtransformersのdevバージョンを必要とします
!pip install -q git+https://github.com/huggingface/transformers
```

## 🖊️ [無料アカウントにサインアップ →](https://app.wandb.ai/login?signup=true)

## 🔑 APIキーを入力
サインアップが完了したら、次のセルを実行してリンクをクリックし、APIキーを取得してノートブックを認証します。


```python
import wandb
wandb.login()
```

オプションで、環境変数を設定してW&Bのログをカスタマイズすることもできます。詳しくは[ドキュメント](https://docs.wandb.com/library/integrations/huggingface)をご覧ください。


```python
# オプション: 勾配とパラメーターの両方をログする
%env WANDB_WATCH=all
```

# 👟 モデルをトレーニング
次に、ダウンロードしたトレーニングスクリプト[run_glue.py](https://huggingface.co/transformers/examples.html#glue)を呼び出し、Weights & Biasesダッシュボードへのトレーニングを自動的にトラッキングします。このスクリプトは、Microsoft Research Paraphrase Corpus (意味的に同等であるかどうかを示す人間の注釈付きの文のペア)にBERTを微調整します。


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
上記のリンクをクリックするか、[wandb.ai](https://app.wandb.ai)にアクセスして結果をリアルタイムで確認してください。依存関係の読み込みが完了すると、ブラウザでrunを表示するためのリンクが表示されます。 "**wandb**: 🚀 View run at [URL to your unique run]" という出力を探してください。

**モデルパフォーマンスの可視化**
数多くの実験を一目で確認し、興味深い発見事項にズームインし、高次元データを可視化するのは簡単です。

![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

**アーキテクチャーの比較**
こちらは[**BERT vs DistilBERT**](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU)の比較例です。自動ラインプロットの可視化により、異なるアーキテクチャーがトレーニング中の評価精度にどのように影響するかを簡単に確認できます。
![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)



### 📈 デフォルトでキー情報を手軽にトラッキング
Weights & Biasesは、実験ごとに新しいrunを保存します。デフォルトで保存される情報は以下のとおりです:
- **ハイパーパラメーター**: Configにモデルの設定が保存されます
- **モデルメトリクス**: メトリクスのストリーミングデータがLogに保存されます
- **端末ログ**: コマンドライン出力が保存され、タブから参照できます
- **システムメトリクス**: GPUやCPUの利用率、メモリ、温度など


## 🤓 詳しく学ぶ
- [ドキュメント](https://docs.wandb.com/huggingface): Weights & BiasesとHugging Faceの統合に関するドキュメント
- [動画](http://wandb.me/youtube): チュートリアル、実務者とのインタビューなど、YouTubeチャンネルでご覧いただけます
- お問い合わせ: contact@wandb.comまでご質問ください