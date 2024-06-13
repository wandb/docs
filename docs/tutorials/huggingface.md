
# Hugging Face

<img src="https://i.imgur.com/vnejHGh.png" width="800"/>

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb)

シームレスな [W&B](https://wandb.ai/site) 統合で、[Hugging Face](https://github.com/huggingface/transformers) モデルの性能をすばやく可視化します。

ハイパーパラメーター、出力メトリクス、GPU利用率などのシステム統計をモデル間で比較しましょう。

## 🤔 なぜW&Bを使うべきなのですか？

<img src="https://wandb.me/mini-diagram" width="650"/>

- **統合ダッシュボード**: すべてのモデルメトリクスと予測のための一元的なリポジトリ
- **軽量**: Hugging Faceと統合するためのコード変更は不要
- **アクセス可能**: 個人および学術チーム向けに無料
- **安全**: すべてのプロジェクトはデフォルトでプライベート
- **信頼性**: OpenAI、トヨタ、Lyftなどの機械学習チームに利用されています

W&Bは、機械学習モデルのためのGitHubのようなものです。機械学習の実験をプライベートにホストされたダッシュボードに保存します。どこでスクリプトを実行しても、すべてのバージョンのモデルが保存されるので、安心して実験できます。

W&Bの軽量な統合は任意のPythonスクリプトで動作し、無料のW&Bアカウントにサインアップするだけで、モデルの追跡と可視化を始めることができます。

Hugging Face Transformersリポジトリでは、トレーナーが各ログステップでトレーニングおよび評価メトリクスを自動的にW&Bにログするように設定されています。

統合がどのように動作するかについての詳細は、こちらをご覧ください: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU)。

# 🚀 インストール、インポート、ログイン

Hugging FaceおよびWeights & Biasesのライブラリ、GLUEデータセット、およびこのチュートリアル用のトレーニングスクリプトをインストールします。
- [Hugging Face Transformers](https://github.com/huggingface/transformers): 自然言語モデルとデータセット
- [Weights & Biases](https://docs.wandb.com/): 実験追跡と可視化
- [GLUE dataset](https://gluebenchmark.com/): 言語理解ベンチマークデータセット
- [GLUE script](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): シーケンス分類用モデルトレーニングスクリプト

```python
!pip install datasets wandb evaluate accelerate -qU
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
```

```python
# run_glue.py スクリプトは transformers dev を必要とします
!pip install -q git+https://github.com/huggingface/transformers
```

## 🖊️ [無料アカウントにサインアップ →](https://app.wandb.ai/login?signup=true)

## 🔑 APIキーを入力
サインアップが完了したら、次のセルを実行してリンクをクリックし、APIキーを取得し、このノートブックを認証します。

```python
import wandb
wandb.login()
```

オプションで、環境変数を設定してW&Bのログをカスタマイズできます。詳しくは [documentation](https://docs.wandb.com/library/integrations/huggingface) を参照してください。

```python
# オプション: 勾配とパラメーターの両方をログ
%env WANDB_WATCH=all
```

# 👟 モデルをトレーニング
次に、ダウンロードしたトレーニングスクリプト [run_glue.py](https://huggingface.co/transformers/examples.html#glue) を実行し、トレーニングがWeights & Biasesのダッシュボードに自動的に追跡されるのを確認します。このスクリプトは、Microsoft Research Paraphrase CorpusにBERTを微調整し、人間の注釈によって意味的に同等かどうかを示す文のペアを使用します。

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
上に印刷されたリンクをクリックするか、[wandb.ai](https://app.wandb.ai) に移動して結果をリアルタイムで確認します。すべての依存関係がロードされた後にブラウザでrunを見るためのリンクが表示されます ― 次の出力を探してください: "**wandb**: 🚀 View run at [URL to your unique run]"

**モデルの性能を可視化**
数十の実験を簡単に見渡し、興味深い発見事項をズームインし、高次元データを可視化するのは簡単です。

![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

**アーキテクチャーの比較**
こちらは[BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU)の比較例です — 自動ラインプロットの可視化で、異なるアーキテクチャーがトレーニング全体を通して評価精度にどのように影響するかを見るのは簡単です。
![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)

### 📈 重要な情報をデフォルトで簡単に追跡
Weights & Biasesは各実験の新しいrunを保存します。デフォルトで保存される情報は次の通りです:
- **ハイパーパラメーター**: Configにモデルの設定が保存されます
- **モデルメトリクス**: メトリクスの時系列データがLogに保存されます
- **端末ログ**: コマンドラインの出力が保存され、タブで利用可能です
- **システムメトリクス**: GPUとCPUの使用率、メモリ、温度など

## 🤓 詳しく学びましょう！
- [Documentation](https://docs.wandb.com/huggingface): Weights & BiasesとHugging Faceの統合に関するドキュメント
- [Videos](http://wandb.me/youtube): チュートリアル、実務者とのインタビューなど、私たちのYouTubeチャンネルで公開中
- Contact: 質問があればcontact@wandb.comまでメッセージを送ってください