---
displayed_sidebar: ja
---
# PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/intro)

PyTorchは、特に研究者の間で、Pythonにおけるディープラーニングのための最も人気のあるフレームワークの一つです。W&Bは、PyTorchに対して、勾配のログ取得からCPUとGPUのコードプロファイリングまで、最高レベルのサポートを提供しています。

:::info
私たちのインテグレーションを[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)でお試しいただくか（ビデオ解説付き）、スクリプトが含まれている[exampleリポジトリ](https://github.com/wandb/examples)をご覧ください。これには、[Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)に[Hyperband](https://arxiv.org/abs/1603.06560)を使用したハイパーパラメータ最適化に関するものが含まれています。さらに、それが生成する[W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)もご覧いただけます。
:::

<!-- {% embed url="https://www.youtube.com/watch?v=G7GH0SeNBMA" %}
ビデオチュートリアルに沿って進めてください！
{% endembed %} -->

## `wandb.watch`を使った勾配の記録

自動的に勾配をログに記録するには、[`wandb.watch`](../../ref/python/watch.md)を呼び出して、PyTorchモデルを渡します。

```python
import wandb

wandb.init(config=args)

model = ...  # モデルの設定

# Magic
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

同じスクリプト内で複数のモデルをトラッキングする必要がある場合は、それぞれのモデルで`wandb.watch`を呼び出すことができます。この関数のリファレンスドキュメントは[こちら](../../ref/python/watch.md)です。

:::caution
勾配、メトリクス、およびグラフは、forwardパスとbackwardパスの後に`wandb.log`が呼び出されるまでログに記録されません。
:::

## 画像やメディアのログ記録

PyTorchの`Tensors`に画像データを渡すと、[`wandb.Image`](../../ref/python/data-types/image.md)と[`torchvision`](https://pytorch.org/vision/stable/index.html)のユーティリティが自動的に画像に変換します。

```python
images_t = ...  # PyTorch Tensorsとして画像を生成または読み込み
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorchや他のフレームワークでW&Bにリッチメディアをログに記録する方法については、[media logging guide](../track/log/media.md)を参照してください。

また、メディアに情報を添えて記録したい場合（モデルの予測や派生したメトリクスなど）、`wandb.Table`を使用してください。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&BにTableをログする
wandb.log({"mnist_predictions": my_table})
```

上のコードでこのようなテーブルが生成されます。このモデルは良さそうです！
![The code above generates a table like this one. This model's looking good!](/images/integrations/pytorch_example_table.png)

データセットとモデルのログや可視化について詳しくは、[W&B Tables のガイド](../tables/intro.md)を参照してください。

## PyTorch コードのプロファイリング

![W&Bダッシュボード内でPyTorchコード実行の詳細トレースを表示。](/images/integrations/pytorch_example_dashboard.png)

W&Bは [PyTorch Kineto](https://github.com/pytorch/kineto)の [Tensorboardプラグイン](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md)と直接統合して、PyTorch コードのプロファイリングツール、CPU と GPU 通信の詳細の検査、ボトルネックの特定や最適化のためのツールを提供しています。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細についてはプロファイラのドキュメントを参照してください
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # ここでプロファイリングしたいコードを実行
    # 詳細な使用方法については、プロファイラのドキュメントを参照してください

# wandbアーティファクトを作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.jsonファイルをアーティファクトに追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# アーティファクトをログに記録
profile_art.save()
```

[このColab](http://wandb.me/trace-colab)で実例コードを確認して実行してください。

:::注意

インタラクティブなトレースビューアツールは、Chrome Trace Viewerをベースにしており、Chromeブラウザで最適に動作します。

:::