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

TensorBoard を既に使用している場合は、wandb と簡単に統合できます。

```python
import tensorflow as tf
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
```

## カスタム メトリクスの ログ記録

TensorBoard に記録されていない追加のカスタム メトリクスを記録する必要がある場合は、コード内で `wandb.log` を呼び出すことができます。`wandb.log({"custom": 0.8})`

Tensorboard の同期時には、`wandb.log` での step 引数の設定はオフになります。異なるステップ数を設定する場合は、次のようにステップ メトリクスと共にメトリクスを記録できます。

``` python
wandb.log({"custom": 0.8, "global_step":global_step}, step=global_step)
```

## TensorFlow Estimators hook

ログに記録される内容をより詳細に制御したい場合、wandb は TensorFlow Estimators 用の hook も提供します。これはグラフ内のすべての `tf.summary` の値をログに記録します。

```python
import tensorflow as tf
import wandb

wandb.init(config=tf.FLAGS)

estimator.train(hooks=[wandb.tensorflow.WandbHook(steps_per_log=1000)])
```

## 手動でログ記録

TensorFlow でメトリクスをログに記録する最も簡単な方法は、TensorFlow ロガーで `tf.summary` をログに記録することです。

```python
import wandb

with tf.Session() as sess:
    # ...
    wandb.tensorflow.log(tf.summary.merge_all())
```

TensorFlow 2 では、カスタム ループでモデルをトレーニングする推奨される方法は、`tf.GradientTape` を使用することです。詳細については、[こちら](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)をご覧ください。カスタム TensorFlow トレーニング ループでメトリクスをログに記録するために `wandb` を組み込む場合は、次のスニペットに従うことができます。

```python
    with tf.GradientTape() as tape:
        # 確率を取得
        predictions = model(features)
        # 損失を計算
        loss = loss_func(labels, predictions)

    # メトリクスをログに記録
    wandb.log("loss": loss.numpy())
    # 勾配を取得
    gradients = tape.gradient(loss, model.trainable_variables)
    # 重みを更新
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

完全な例は[こちら](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2)にあります。

## W&B は TensorBoard とどう違うのですか？

共同創業者が W&B の開発を始めたとき、OpenAI の不満を抱えた TensorBoard ユーザーのために ツール を構築することに触発されました。改善に重点を置いている点をいくつかご紹介します。

1.  **モデル の再現**: Weights & Biases は、実験、探索、および後で モデル を再現するのに適しています。メトリクスだけでなく、ハイパーパラメーター と コード の バージョン もキャプチャし、プロジェクト を再現可能にするために、バージョン管理のステータスと モデル の チェックポイント を保存できます。
2.  **自動的な整理**: コラボレーター から プロジェクト を引き継ぐ場合でも、休暇から戻ってきた場合でも、古い プロジェクト を再開する場合でも、W&B を使用すると、試行されたすべての モデル を簡単に確認できるため、誰も時間、GPU サイクル、または二酸化炭素を浪費して 実験 を再実行することはありません。
3.  **高速で柔軟な インテグレーション**: 5 分で W&B を プロジェクト に追加します。無料のオープンソース Python パッケージ をインストールし、数行の コード を追加するだけで、モデル を実行するたびに、優れたログ記録された メトリクス と レコード が得られます。
4.  **永続的な集中 ダッシュボード**: ローカル マシン、共有 ラボ クラスター、または クラウド の スポット インスタンス のいずれで モデル を トレーニング しても、結果は同じ集中 ダッシュボード に共有されます。異なる マシン から TensorBoard ファイル をコピーして整理するのに時間を費やす必要はありません。
5.  **強力な テーブル**: さまざまな モデル からの結果を検索、フィルタリング、ソート、およびグループ化します。数千の モデル バージョン を調べて、さまざまな タスク に最適な パフォーマンス の モデル を簡単に見つけることができます。TensorBoard は大規模な プロジェクト でうまく機能するように構築されていません。
6.  **コラボレーション のための ツール**: W&B を使用して、複雑な 機械学習 プロジェクト を整理します。W&B へのリンクを簡単に共有でき、プライベート Teams を使用して、全員が結果を共有 プロジェクト に送信できます。レポート を介した コラボレーション もサポートしています。インタラクティブな 可視化 を追加し、markdown で作業内容を記述します。これは、作業ログを保持したり、上司と学びを共有したり、ラボや チーム に学びを提示したりするのに最適な方法です。

[無料 アカウント](https://wandb.ai) から始めましょう

## 例

インテグレーション がどのように機能するかを確認するためのいくつかの例を作成しました。

*   [Github の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-estimator-mnist/mnist.py): TensorFlow Estimators を使用した MNIST の例
*   [Github の例](https://github.com/wandb/examples/blob/master/examples/tensorflow/tf-cnn-fashion/train.py): 生の TensorFlow を使用した Fashion MNIST の例
*   [Wandb ダッシュボード](https://app.wandb.ai/l2k2/examples-tf-estimator-mnist/runs/p0ifowcb): W&B での結果を表示
*   TensorFlow 2 での トレーニング ループ のカスタマイズ - [記事](https://www.wandb.com/articles/wandb-customizing-training-loops-in-tensorflow-2) | [ダッシュボード](https://app.wandb.ai/sayakpaul/custom_training_loops_tf)
