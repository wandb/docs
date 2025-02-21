---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、特に研究者の間で、Python におけるディープラーニングの最も人気のあるフレームワークのひとつです。W&B は、勾配のログから CPU や GPU 上でのコードのプロファイリングまで、PyTorch に対して第一級のサポートを提供します。

Colab ノートブックで私たちのインテグレーションを試してみてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

また、[example repo](https://github.com/wandb/examples) でスクリプトを確認できます。[Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) 上で [Hyperband](https://arxiv.org/abs/1603.06560) を使用したハイパーパラメーター最適化を含みます。また、生成される [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) もご覧ください。

## `wandb.watch` で勾配をログする

勾配を自動でログするには、[`wandb.watch`]({{< relref path="/ref/python/watch.md" lang="ja" >}}) を呼び出し、PyTorch モデルを渡します。

```python
import wandb

wandb.init(config=args)

model = ...  # モデルを設定

# 魔法
wandb.watch(model, log_freq=100)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```

同じスクリプト内で複数のモデルをトラッキングする必要がある場合、各モデルに対して `wandb.watch` を個別に呼び出すことができます。この関数のリファレンスドキュメントは[こちら]({{< relref path="/ref/python/watch.md" lang="ja" >}})です。

{{% alert color="secondary" %}}
勾配、メトリクス、およびグラフは、フォワードおよびバックワードパスの後に `wandb.log` が呼び出されるまでログされません。
{{% /alert %}}

## 画像とメディアをログする

画像データを持つ PyTorch `Tensors` を [`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}}) に渡すことで、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティを使って自動的に画像に変換されます：

```python
images_t = ...  # PyTorch Tensors として画像を生成またはロードする
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch や他のフレームワークでリッチメディアを W&B にログする方法についての詳細は、[メディアログガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})をご覧ください。

メディアに付随する情報を含めたい場合は、モデルの予測や派生メトリクスなどを `wandb.Table` を利用して追加できます。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# テーブルを W&B にログする
wandb.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="上記のコードはこのようなテーブルを生成します。このモデルは良好です！" >}}

データセットとモデルのログと可視化についての詳細は、[W&B テーブル ガイド]({{< relref path="/guides/core/tables/" lang="ja" >}})をご覧ください。

## PyTorch コードをプロファイルする

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="W&B ダッシュボード内で PyTorch コード実行の詳細なトレースを表示する。" >}}

W&B は [PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard プラグイン](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接連携し、PyTorch コードのプロファイリング、CPU と GPU の通信の詳細な検査、ボトルネックや最適化の特定のためのツールを提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジュールに関する詳細はプロファイラのドキュメントを参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # ここでプロファイルしたいコードを実行する
    # 詳細な使用法についてはプロファイラのドキュメントを参照

# wandb Artifact を作成する
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルを Artifact に追加する
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# アーティファクトをログする
profile_art.save()
```

[この Colab](http://wandb.me/trace-colab) で動作するサンプルコードを確認し、実行してみてください。

{{% alert color="secondary" %}}
インタラクティブなトレース表示ツールは、Chrome ブラウザで最適に動作する Chrome Trace Viewer に基づいています。
{{% /alert %}}