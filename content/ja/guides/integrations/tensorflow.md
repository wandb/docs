---
title: TensorFlow
menu:
  default:
    identifier: ja-guides-integrations-tensorflow
    parent: integrations
weight: 440
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM" >}}

## 始めましょう

すでに TensorBoard を使用している場合は、wandb と簡単に統合できます。

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## カスタムメトリクスをログする

TensorBoard にログされていない追加のカスタムメトリクスをログする必要がある場合は、`wandb.log` をコード内で呼び出すことができます `wandb.log({"custom": 0.8})`

`wandb.log` でステップ引数を設定することは TensorBoard の同期時にオフになっています。異なるステップカウントを設定したい場合は、次のようにステップメトリクスとしてメトリクスをログできます：

``` python
wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow estimators フック

ログする内容をより制御したい場合、wandb も TensorFlow estimators 用のフックを提供しています。これによりグラフ内のすべての `tf.summary` 値がログされます。

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 手動でログする

TensorFlow でメトリクスをログする最も簡単な方法は、TensorFlow ロガーで `tf.summary` をログすることです：

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタムループでモデルをトレーニングするための推奨方法は `tf.GradientTape` を使用することです。詳細はこちら [here](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) で読むことができます。カスタム TensorFlow トレーニングループに `wandb` を組み込んでメトリクスをログしたい場合は、次のスニペットを参照してください：

```python
    with tf.GradientTape() as tape:
        # 確率を取得する
        predictions = model(features)
        # 損失を計算する
        loss = loss_func(labels, predictions)

    # メトリクスをログする
    wandb.log("loss": loss.numpy())
    # 勾配を取得する
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新する
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

完全な例は [here](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) で利用可能です。

## W&B は TensorBoard と何が違うのですか？

共同創設者たちは W&B の開発を始めたとき、OpenAI のフラストレーションを感じている TensorBoard ユーザーのためのツールを作ることをインスパイアされました。私たちが改善に集中したいくつかの点を紹介します：

1. **モデルの再現性**: Weights & Biases は実験、探索、および後でモデルを再現することに優れています。メトリクスだけでなく、ハイパーパラメーターやコードのバージョンもキャプチャし、バージョン管理の状態やモデルのチェックポイントを保存することができるため、プロジェクトが再現可能です。
2. **自動整理**: 協力者からプロジェクトを引き継ぐ場合、休暇から戻ってくる場合、または古いプロジェクトを見直す場合、W&B は試されたすべてのモデルを簡単に表示でき、誰も何時間、GPU サイクル、または二酸化炭素を無駄に再実行することがないようにします。
3. **高速で柔軟なインテグレーション**: 5 分でプロジェクトに W&B を追加します。無料のオープンソース Python パッケージをインストールして、コードに数行追加するだけで、モデルを実行するたびにログされたメトリクスと記録がきちんと表示されます。
4. **永続的で集中管理されたダッシュボード**: モデルをローカル マシン、共有ラボ クラスター、またはクラウドのスポットインスタンスでトレーニングするかどうかにかかわらず、結果は同じ集中ダッシュボードに共有されます。異なるマシンから TensorBoard ファイルをコピーして整理するのに時間を費やす必要はありません。
5. **強力なテーブル**: 異なるモデルの結果を検索、フィルター、並べ替え、およびグループ化します。数千のモデルバージョンを簡単に見渡し、さまざまなタスクに最も適したモデルを見つけることができます。TensorBoard は大規模なプロジェクトでうまく機能するように設計されていません。
6. **コラボレーションのためのツール**: 複雑な機械学習プロジェクトを整理するために W&B を使用します。W&B へのリンクを簡単に共有でき、プライベートチームを使用して全員が共有プロジェクトに結果を送信することができます。また、対話型の可視化を追加し、markdown で作業内容を説明することで、Reports を通じたコラボレーションもサポートしています。これは作業ログを保持したり、上司と学びを共有したり、ラボまたはチームに学びを発表するのに最適な方法です。

[無料アカウント](https://wandb.ai) で始めましょう

## 例

インテグレーションがどのように機能するかを見るためにいくつかの例を作成しました：

* [Github 上の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使用した MNIST の例
* [Github 上の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): Raw TensorFlow を使用した Fashion MNIST の例
* [Wandb ダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B での結果を見る
* TensorFlow 2 でトレーニングループをカスタマイズする - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)