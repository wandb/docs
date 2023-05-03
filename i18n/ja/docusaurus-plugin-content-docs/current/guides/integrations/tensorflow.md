# TensorFlow

TensorBoardをすでに使用している場合、wandbとの統合も簡単です。

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## カスタムメトリクス

TensorBoardにログされていない追加のカスタムメトリクスをログする必要がある場合は、コード内で `wandb.log` を呼び出すことができます。`wandb.log({"custom": 0.8}) `

TensorBoardを同期する際、`wandb.log`のstep引数の設定は無効になります。別のステップ数を設定したい場合は、ステップメトリクスとともにメトリクスを記録できます。

`wandb.log({"custom": 0.8, "global_step"=global_step})`

## TensorFlowフック

ログしたい内容に対してもっとコントロールしたい場合、wandbはTensorFlowの推定器（estimators）用のフックも提供しています。グラフ内のすべての`tf.summary`値がログされます。

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```
## 手動ログ

TensorFlowでメトリクスをログに記録する最も簡単な方法は、TensorFlowロガーで`tf.summary`をログに記録することです。

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2では、カスタムループでモデルをトレーニングする推奨される方法は、`tf.GradientTape`を使用することです。詳しくは[こちら](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)をご覧ください。カスタムTensorFlowトレーニングループで`wandb`を使用してメトリクスをログに記録する方法は、以下のスニペットに従ってください。

```python
    with tf.GradientTape() as tape:
        # 確率を取得
        predictions = model(features)
        # 損失を計算
        loss = loss_func(labels, predictions)

    # メトリクスを記録
    wandb.log({"loss": loss.numpy()})
    # 勾配を取得
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

完全な例は[こちら](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)で利用できます。
## W&BはTensorBoardとどのように違いますか？

W&Bの共同創設者が開発を始めたとき、彼らはOpenAIのTensorBoardユーザーのためのツールを作ることにインスパイアされました。以下は、私たちが改善に取り組んできたいくつかの点です。

1. **モデルの再現性**: Weights & Biasesは実験、探索、そして後でモデルを再現するのに適しています。メトリクスだけでなく、ハイパーパラメータやコードのバージョンもキャプチャし、バージョン管理の状況やモデルのチェックポイントを保存してプロジェクトを再現可能にします。

2. **自動的な整理**: 他の人からプロジェクトを引き継いだり、休暇から戻ったり、古いプロジェクトを掘り出したりする場合、W&Bを使えばこれまで試したすべてのモデルが簡単に表示されるため、誰もが実験を再実行して時間やGPUサイクル、炭素を無駄にすることはありません。

3. **高速で柔軟な統合**: W&Bを5分でプロジェクトに追加します。無料のオープンソースPythonパッケージをインストールし、コードに2行加えるだけで、モデルを実行するたびにすばらしいログ付きメトリクスとレコードが手に入ります。

4. **永続的で集中化されたダッシュボード**: モデルをどこでトレーニングしても、ローカルマシン、共有ラボクラスター、クラウドのスポットインスタンスなど、結果は同じ集中ダッシュボードに共有されます。異なるマシンからTensorBoardファイルをコピーして整理する時間を費やす必要はありません。

5. **強力なテーブル**: 異なるモデルからの結果を検索、フィルタ、ソート、グループ化します。何千ものモデルバージョンを一目で確認し、さまざまなタスクに最適なモデルを見つけることが簡単です。TensorBoardは大規模なプロジェクトでうまく動作するようには設計されていません。

6. **コラボレーションのためのツール**: W&Bを使って複雑な機械学習プロジェクトを整理します。W&Bへのリンクを簡単に共有でき、プライベートチームを使ってみんなが共有プロジェクトに結果を送信できます。また、インタラクティブな可視化を追加したり、マークダウンで作業内容を説明したりすることで、レポートを通じたコラボレーションもサポートしています。これは、作業ログを残したり、上司との連絡を共有したり、研究室やチームに結果を報告するのに最適な方法です。

[無料の個人アカウントで始める →](https://wandb.ai)

## 例

以下は、統合方法を示すいくつかの例です。

* [GitHubの例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimatorsを使用したMNISTの例

* [GitHubの例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlowを使用したFashion MNISTの例

* [Wandbダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&Bでの結果表示

* TensorFlow 2でのトレーニングループのカスタマイズ - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [Colabノートブック](https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)