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

すでに TensorBoard を利用している場合は、wandb とのインテグレーションはとても簡単です。

```python
import tensorflow as tf
import wandb
```

## カスタムメトリクスのログ

TensorBoard にまだ記録されていない独自のカスタムメトリクスを記録したい場合は、`run.log()` をコード内で呼び出すことができます。  
`run.log({"custom": 0.8})` のように使います。

Tensorboard 同期時には `run.log()` の step 引数は自動的に無効になります。もし異なる step 数を設定したい場合は、以下のようにステップ付きでメトリクスを記録してください。

``` python
with wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True) as run:
    run.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow Estimators フック

ログする内容をさらに細かく制御したい場合は、wandb は TensorFlow Estimators 用のフックも提供しています。これを使うと、グラフ内のすべての `tf.summary` の値がログされます。

```python
import tensorflow as tf
import wandb

run = wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
run.finish()
```

## 手動でログする

TensorFlow でメトリクスをログする最もシンプルな方法は、TensorFlow の logger を使って `tf.summary` を記録することです。

```python
import wandb
run = wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、独自ループでモデルをトレーニングする際には `tf.GradientTape` を使うのが推奨されています。詳細は [TensorFlow カスタムトレーニングウォークスルー](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) をご覧ください。独自の TensorFlow トレーニングループでメトリクスを wandb で記録したい場合、以下のサンプルを参考にできます。

```python
    with tf.GradientTape() as tape:
        # 確率を取得
        predictions = model(features)
        # ロスを計算
        loss = loss_func(labels, predictions)

    # メトリクスをログ
    run.log("loss": loss.numpy())
    # 勾配を取得
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

[TensorFlow 2 のトレーニングループをカスタマイズしたフル例](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) も用意しています。

## W&B は TensorBoard と何が違うの？

W&B の共同創業者たちは、OpenAI で TensorBoard 利用時に感じたフラストレーションをきっかけにこのツールの開発を始めました。W&B は以下の点を重視して改良しています。

1. **モデルの再現性**: W&B は実験・探索や後のモデル再現に適しています。メトリクスだけでなく、ハイパーパラメーターやコードのバージョンもキャプチャし、バージョン管理の状態やモデルのチェックポイントも保存できるため、プロジェクトの再現が簡単です。
2. **自動整理**: 協力者からプロジェクトを引き継ぐときや、休暇明け・古いプロジェクトを再開する際でも、W&B では試行されたすべてのモデルが簡単に見えるため、無駄な時間や GPU 計算、実験の繰り返しを防げます。
3. **高速で柔軟なインテグレーション**: W&B の導入は 5 分で完了します。無料の OSS パッケージをインストールして数行コードを加えるだけで、モデルを実行するたびにメトリクスや記録が自動で残ります。
4. **永続的かつ一元化されたダッシュボード**: ローカル PC、共有ラボクラスター、クラウド・スポットインスタンスなど、どこでモデルをトレーニングしても結果は同じダッシュボードで一元管理されます。異なるマシンから TensorBoard のファイルをコピー・整理する手間は不要です。
5. **強力なテーブル**: 異なるモデルの結果を検索・フィルタ・ソート・グルーピング可能です。大量のモデルバージョンの中から、用途別に最もパフォーマンスの良いモデルを素早く見つけることができます。TensorBoard は大規模プロジェクトでうまく機能しません。
6. **コラボレーション向けのツール**: W&B を使えば、複雑な機械学習プロジェクトも整理できます。W&B へのリンクを共有したり、プライベートチームでみんなの結果を同じプロジェクトへ送信できます。さらに、Reports を使ってインタラクティブな可視化や説明を書けるので、進捗ログの維持や上司への共有、ラボやチームへの成果発表にも最適です。

[無料アカウント](https://wandb.ai) で今すぐ始めましょう

## 例

以下に wandb とのインテグレーション例をいくつか紹介します。

* [Github 上の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使った MNIST の例
* [Github 上の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 生の TensorFlow での Fashion MNIST の例
* [Wandb ダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B での結果を表示
* TensorFlow 2 でのトレーニングループカスタマイズ  
  - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
