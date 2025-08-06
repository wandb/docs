---
title: TensorFlow
menu:
  default:
    identifier: tensorflow
    parent: integrations
weight: 440
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1JCpAbjkCFhYMT7LCQ399y35TS3jlMpvM" >}}

## はじめに

すでに TensorBoard を使っている場合、簡単に wandb とのインテグレーションが可能です。

```python
import tensorflow as tf
import wandb
```

## カスタムメトリクスのログ

TensorBoard では記録していない追加のカスタムメトリクスをログしたい場合、`run.log()` をコード内で呼び出してください。  
例: `run.log({"custom": 0.8})`

TensorBoard の同期中は `run.log()` の step 引数の設定は無効になります。別の step 値でメトリクスをログしたい場合は、次のように step メトリクス付きでログできます。

``` python
with wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True) as run:
    run.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow Estimator 用のフック

ログ内容をさらに細かく制御したい場合は、wandb は TensorFlow Estimator 用のフックも提供しています。これにより、グラフ内のすべての `tf.summary` の値をログします。

```python
import tensorflow as tf
import wandb

run = wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
run.finish()
```

## 手動でログする

TensorFlow でメトリクスをログする最もシンプルな方法は、TensorFlow のロガーで `tf.summary` をログすることです。

```python
import wandb
run = wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタムループでモデルをトレーニングする推奨方法は `tf.GradientTape` を使うことです。詳細は [TensorFlow のカスタムトレーニングウォークスルー](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) をご覧ください。  
独自の TensorFlow トレーニングループに wandb を組み込んでメトリクスをログしたい場合、以下のように記述できます。

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

[TensorFlow 2 でトレーニングループをカスタマイズする完全な例](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) も用意しています。

## W&B と TensorBoard の違いは？

W&B の共同創業者たちがプロダクト開発を始めたとき、OpenAI でフラストレーションを感じていた TensorBoard ユーザーのためのツール作りを目指しました。私たちが特に改善に注力したポイントをいくつかご紹介します。

1. **モデルの再現性**: W&B は実験や探索、後からモデルを再現するのに最適です。メトリクスだけでなく、ハイパーパラメーター、コードのバージョンも記録し、バージョン管理の状態やモデルチェックポイントも保存できます。これによりプロジェクトの再現性が高まります。
2. **自動整理**: コラボレーターから引き継ぐ場合や、長い休暇明け、古いプロジェクトを再開する際でも、W&B ならどのモデルが試されたか一目で把握できます。誰も無駄な時間や GPU リソース、カーボンを消費して同じ実験を繰り返す必要はありません。
3. **高速かつ柔軟なインテグレーション**: 5分で W&B をプロジェクトに追加できます。無料のオープンソース Python パッケージをインストールし、数行コードを加えるだけで、モデルを実行するたびに綺麗なメトリクスと記録が残ります。
4. **永続的かつ集中管理されたダッシュボード**: どこでモデルをトレーニングしても、ローカルマシン、共同研究室のクラスター、クラウド上のスポットインスタンスでも、結果はすべて同じ中央ダッシュボードに共有されます。TensorBoard のファイルを各マシンからコピー・整理する手間は不要です。
5. **高機能テーブル**: 検索、フィルタ、ソート、グループ化で、異なるモデルの結果を一覧できます。数千ものモデルバージョンを比較し、さまざまなタスクで最も性能の良いモデルを簡単に探せます。TensorBoard は大規模プロジェクトには向いていません。
6. **コラボレーションのためのツール**: W&B を使えば複雑な機械学習プロジェクトも整理できます。W&B のリンク共有は簡単で、プライベートチームを使えば全員が同じプロジェクトに結果を送信可能です。レポートによるコラボレーションにも対応しています。インタラクティブな可視化を加えたり、markdown で作業内容の記述もできます。これにより開発ログを残したり、上司と学びを共有したり、ラボやチームへの発表にも最適です。

[無料アカウントで始める](https://wandb.ai)

## サンプル

W&B とのインテグレーション方法が分かるサンプルをいくつかご用意しました。

* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimator を使った MNIST の例
* [Example on Github](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 生の TensorFlow を使った Fashion MNIST の例
* [Wandb Dashboard](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B で結果を表示
* TensorFlow 2 のトレーニングループカスタマイズ - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
