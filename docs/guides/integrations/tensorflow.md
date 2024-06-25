---
displayed_sidebar: default
---


# TensorFlow

すでに TensorBoard を使っているなら、wandb とのインテグレーションは簡単です。

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## カスタムメトリクス

TensorBoard にログされない追加のカスタムメトリクスをログする必要がある場合は、コード内で `wandb.log` を呼び出すことができます。 `wandb.log({"custom": 0.8})`

Tensorboard と同期する際には、`wandb.log` でステップ引数を設定することはできません。異なるステップカウントを設定したい場合は、以下のようにステップメトリクスを使用してメトリクスをログできます。

`wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)`

## TensorFlow Hook

ログする内容をより細かく制御したい場合、wandb は TensorFlow エスティメーター用のフックも提供しています。これにより、グラフ内のすべての `tf.summary` 値をログすることができます。

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 手動でのログ

TensorFlow でメトリクスをログする最も簡単な方法は、TensorFlow ロガーで `tf.summary` をログすることです。

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタムループでモデルをトレーニングする推奨方法は `tf.GradientTape` を使用することです。詳細は[こちら](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)で読むことができます。カスタム TensorFlow トレーニングループに `wandb` を組み込んでメトリクスをログしたい場合は、以下のスニペットを参考にしてください:

```python
    with tf.GradientTape() as tape:
        # 確率を取得
        predictions = model(features)
        # 損失を計算
        loss = loss_func(labels, predictions)

    # メトリクスをログ
    wandb.log("loss": loss.numpy())
    # 勾配を取得
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

完全な例は[こちら](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)にあります。

## W&B と TensorBoard の違いは？

共同創業者が W&B を開発し始めたとき、OpenAI のフラストレーションを抱える TensorBoard ユーザーのためのツールを作ることにインスパイアされました。私たちが改善に力を入れた点をいくつか紹介します。

1. **モデルの再現性**: Weights & Biases は実験、探究、そして後からのモデル再現に優れています。メトリクスだけでなく、ハイパーパラメーターやコードのバージョンもキャプチャし、バージョン管理のステータスやモデルチェックポイントも保存できるので、プロジェクトの再現性が高まります。
2. **自動整理**: 協力者からプロジェクトを引き継ぐとき、休暇から戻ったとき、古いプロジェクトを見返すときなど、W&B はすべての試行モデルを簡単に確認できるようにします。これにより時間、GPU サイクル、実験の再実行による二酸化炭素の無駄を避けることができます。
3. **高速かつ柔軟なインテグレーション**: W&B をプロジェクトに追加するのに 5 分もかかりません。私たちの無料オープンソース Python パッケージをインストールし、コードに数行追加するだけで、モデルを走らせるたびに素晴らしいログメトリクスと記録が得られます。
4. **永続的で集中化されたダッシュボード**: ローカルマシン、共有作業用クラスター、クラウドのスポットインスタンスなど、どこでモデルをトレーニングしても、結果は同じ集中化されたダッシュボードに共有されます。異なるマシンから TensorBoard ファイルをコピーして整理する時間を節約できます。
5. **強力なテーブル**: 異なるモデルからの結果を検索、フィルタリング、ソート、グループ化できます。何千ものモデルバージョンを調べ、異なるタスクに最適なモデルを簡単に見つけることができます。TensorBoard は大規模プロジェクトには適していません。
6. **コラボレーションツール**: W&B を使用して複雑な機械学習プロジェクトを整理します。W&B へのリンクを共有するのは簡単で、プライベートチームを使用して全員が結果を共有プロジェクトに送信できるようにすることもできます。Reports を通じたコラボレーションもサポートしています。インタラクティブな可視化を追加し、markdown で作業内容を記述できます。これは、作業ログの保持、上司との学びの共有、ラボやチームへの成果発表に最適です。

[無料アカウント](https://wandb.ai)で始めましょう。

## 例

インテグレーションがどのように機能するかを示すいくつかの例を作成しました：

* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使用した MNIST の例
* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow を使用した Fashion MNIST の例
* [Wandb Dashboard](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B 上で結果を表示
* TensorFlow 2 でのトレーニングループのカスタマイズ - [Article](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [Colab Notebook](https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM) | [Dashboard](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)