---
title: TensorFlow
menu:
  default:
    identifier: ja-guides-integrations-tensorflow
    parent: integrations
weight: 440
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM" >}}

## はじめに

TensorBoard をすでに使用している場合、wandb との連携は簡単です。

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## カスタム メトリクスのログ

TensorBoard に記録されていない追加のカスタム メトリクスを記録する必要がある場合は、コード内で `wandb.log` を呼び出すことができます。`wandb.log({"custom": 0.8})`

Tensorboard を同期すると、`wandb.log` の step 引数の設定はオフになります。別のステップ数を設定する場合は、次のようにステップ メトリクスを使用してメトリクスを記録できます。

``` python
wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow estimators hook

ログに記録する内容をより詳細に制御したい場合は、wandb は TensorFlow estimators 用の hook も提供します。グラフ内のすべての `tf.summary` の値を記録します。

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 手動でのログ

TensorFlow でメトリクスをログに記録する最も簡単な方法は、TensorFlow ロガーで `tf.summary` をログに記録することです。

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタム ループでモデルをトレーニングするための推奨される方法は、`tf.GradientTape` を使用することです。詳細については、[こちら](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)をご覧ください。`wandb` を組み込んでカスタム TensorFlow トレーニング ループでメトリクスをログに記録する場合は、次のスニペットに従ってください。

```python
    with tf.GradientTape() as tape:
        # Get the probabilities
        predictions = model(features)
        # Calculate the loss
        loss = loss_func(labels, predictions)

    # Log your metrics
    wandb.log("loss": loss.numpy())
    # Get the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

完全な例は[こちら](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)にあります。

## W&B と TensorBoard の違いは何ですか?

共同創設者が W&B の開発を開始したとき、OpenAI の不満を抱えた TensorBoard ユーザーのためにツールを構築することに触発されました。改善に重点を置いている点をいくつかご紹介します。

1. **モデルの再現**: Weights & Biases は、実験 、探索、および後でモデルを再現するのに適しています。メトリクスだけでなく、ハイパーパラメーターとコードのバージョンもキャプチャし、プロジェクトを再現できるように、バージョン管理ステータスとモデルのチェックポイントを保存できます。
2. **自動編成**: コラボレーターからプロジェクトを引き継ぐ場合でも、休暇から戻ってきた場合でも、古いプロジェクトを整理する場合でも、W&B を使用すると、試行されたすべてのモデルを簡単に確認できるため、誰も時間、 GPU サイクル、またはカーボンを無駄に実験を再実行することはありません。
3. **高速で柔軟なインテグレーション**: W&B を 5 分でプロジェクトに追加します。無料のオープンソース Python パッケージをインストールし、コードに数行追加するだけで、モデルを実行するたびに、適切なログに記録されたメトリクスとレコードが得られます。
4. **永続的で集中化されたダッシュボード**: ローカル マシン、共有ラボ クラスター、クラウドのスポット インスタンスなど、モデルをどこでトレーニングする場合でも、結果は同じ集中化されたダッシュボードに共有されます。さまざまなマシンから TensorBoard ファイルをコピーして整理する時間を費やす必要はありません。
5. **強力な テーブル**: さまざまなモデルの結果を検索、フィルタリング、ソート、およびグループ化します。数千のモデル バージョンを確認し、さまざまなタスクに最適なモデルを簡単に見つけることができます。TensorBoard は、大規模なプロジェクトでうまく機能するように構築されていません。
6. **コラボレーション ツール**: W&B を使用して、複雑な 機械学習 プロジェクトを整理します。W&B へのリンクを簡単に共有でき、プライベート Teams を使用して、全員が結果を共有プロジェクトに送信できます。Reports を介したコラボレーションもサポートしています。インタラクティブな 可視化を追加し、markdown で作業内容を記述します。これは、作業ログを保持し、上司と学びを共有したり、ラボや Teams に学びを提示したりするのに最適な方法です。

[無料アカウント](https://wandb.ai)を始めましょう

## 例

インテグレーションの仕組みを示すために、いくつかの例を作成しました。

* [Github の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使用した MNIST の例
* [Github の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 生の TensorFlow を使用した Fashion MNIST の例
* [Wandb ダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B で結果を表示
* TensorFlow 2 でのトレーニング ループのカスタマイズ - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
