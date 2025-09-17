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

すでに TensorBoard を使っているなら、wandb との統合は簡単です。

```python
import tensorflow as tf
import wandb
```

## カスタム メトリクスをログする

TensorBoard に記録されていない追加のカスタム メトリクスをログしたい場合は、コード内で `run.log()` を呼び出せます。例: `run.log({"custom": 0.8})`

TensorBoard と同期しているときは、`run.log()` の step 引数の設定は無効になります。別のステップ数を設定したい場合は、次のように step メトリクス付きでメトリクスをログできます:

``` python
with wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True) as run:
    run.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow Estimator のフック

ログ内容をより細かく制御したい場合、wandb は TensorFlow Estimator 向けの フック も提供しています。グラフ内の `tf.summary` の値をすべてログします。

```python
import tensorflow as tf
import wandb

run = wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
run.finish()
```

## 手動でログする

TensorFlow でメトリクスをログする最も簡単な方法は、TensorFlow のロガーで `tf.summary` を記録することです:

```python
import wandb
run = wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタム ループで モデル を学習する推奨方法は `tf.GradientTape` を使うことです。詳しくは [TensorFlow custom training walkthrough](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) を参照してください。カスタムの TensorFlow トレーニング ループに wandb を組み込んでメトリクスをログしたい場合は、次のスニペットに従ってください:

```python
    with tf.GradientTape() as tape:
        # 確率を取得
        predictions = model(features)
        # 損失を計算
        loss = loss_func(labels, predictions)

    # メトリクスをログする
    run.log("loss": loss.numpy())
    # 勾配を取得
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

[TensorFlow 2 でトレーニング ループをカスタマイズする完全な例](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) も用意しています。

## W&B は TensorBoard とどう違いますか？

共同創業者が W&B を作り始めたとき、OpenAI のフラストレーションを抱えた TensorBoard ユーザーのための ツール を作ることにインスパイアされました。以下は、私たちが改善に注力してきた点です。

1.  **モデルの再現**: W&B は実験、探索、そして後から モデル を再現するのに適しています。メトリクスだけでなく、ハイパーパラメーターや コード のバージョンも取得し、バージョン管理の状態や モデル のチェックポイントも保存できるため、プロジェクトの再現性が高まります。
2. **自動的な整理**: 共同研究者のプロジェクトを引き継ぐとき、休暇から戻ったとき、昔のプロジェクトを掘り起こすときでも、W&B なら試された モデル をすべて簡単に把握できます。これにより、誰も余計な時間や GPU サイクル、二酸化炭素排出をかけて 実験 を再実行してしまうことがありません。
3. **高速で柔軟なインテグレーション**: 5 分で W&B をあなたのプロジェクトに追加できます。無料のオープンソース Python パッケージをインストールして コード に数行追加するだけで、モデルを実行するたびにメトリクスや記録がきれいにログされます。
4. **永続的で一元化された ダッシュボード**: モデル をどこで学習しても、ローカル マシンでも、共有ラボの クラスター でも、クラウド のスポットインスタンスでも、結果は同じ一元化された ダッシュボード に共有されます。異なるマシンから TensorBoard のファイルをコピーして整理することに時間を費やす必要はありません。
5. **強力なテーブル**: 異なる モデル の結果を検索、フィルタ、ソート、グループ化できます。何千もの モデル バージョンを俯瞰して、タスクごとに最も性能の良い モデル を見つけるのが簡単です。TensorBoard は大規模なプロジェクトでうまく機能するようには作られていません。
6. **コラボレーションのための ツール**: W&B で複雑な 機械学習 プロジェクトを整理できます。W&B のリンクを共有するのは簡単で、プライベート Teams を使って全員が結果を共有のプロジェクトに送ることができます。さらに Reports を使ったコラボレーションもサポートしています — インタラクティブな可視化を追加し、Markdown で作業内容を説明できます。作業ログを残したり、上長と学びを共有したり、ラボやチームに結果を発表したりするのに最適です。

[無料アカウントで始める](https://wandb.ai)

## 例

統合がどのように機能するかを確認できるよう、いくつか例を用意しました:

* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使った MNIST の例
* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 素の TensorFlow を使った Fashion MNIST の例
* [W&B ダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B で結果を見る
* TensorFlow 2 でのトレーニング ループのカスタマイズ - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)