# PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/intro)

PyTorchは、Pythonでのディープラーニング用の非常に一般的なフレームワークの1つで、特に研究者がよく使用しています。W&Bは、CPUとGPU上での勾配のロギングからコードのプロファイリングまで、PyTorchの一流のサポートを提供します。

:::info
[colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb) (with video walkthrough below) or see our [example repo](https://github.com/wandb/examples) for scripts, including one on hyperparameter optimization using [Hyperband](https://arxiv.org/abs/1603.06560) on [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion), plus the [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) it generates.
:::

:::info
[colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb)（以下の動画ガイドを参照）で統合を試したり、スクリプト用の[レポジトリ例](https://github.com/wandb/examples)（[Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)で[Hyperband](https://arxiv.org/abs/1603.06560)を使用したハイパーパラメーターの最適化に関する例を含む）や、生成される[W&Bダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs?workspace=)を確認したりしましょう
:::

<!-- {% embed url="https://www.youtube.com/watch?v=G7GH0SeNBMA" %}
Follow along with a video tutorial!
{% endembed %} -->

## wandb.watchによる勾配のロギング​

自動的に勾配を記録するには、[`wandb.watch`](../../ref/python/watch.md)を呼び出してPyTorchモデルで渡すことができます。

```python
import wandb
wandb.init(config=args)

model = ... # set up your model

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

同じスクリプトで複数のモデルをトラッキングする必要がある場合は、各モデルで「wandb.watch」を個別に呼び出すことができます。この関数に関するドキュメントを[こちら](../../ref/python/watch.md)でご覧ください。

:::caution
フォワードパスまたはバックワードパスの後に「wandb.log」が呼び出されるまで、勾配、メトリクスおよびグラフは記録されません。
:::

## 画像とメディアのロギング​

PyTorch Tensorsを画像データとともに[`wandb.Image`](../../ref/python/data-types/image.md)に渡すことができます。[`torchvision`](https://pytorch.org/vision/stable/index.html) からのユーティリティを使って、これらが自動的に画像に変換されます：

```python
images_t = ...  # generate or load images as PyTorch Tensors
wandb.log({"examples" : [wandb.Image(im) for im in images_t]})
```

PyTorchとその他のフレームワークでの、リッチメディアのW&Bへのロギングについての詳細は、[メディアロギングガイド](../track/log/media.md)をご覧ください。

モデルの予測や抽出されたメトリクスなどのメディアと共に情報を含めたい場合は、「wandb.Table」を使用します。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# Log your Table to W&B
wandb.log({"mnist_predictions": my_table})
```

![The code above generates a table like this one. This model's looking good!](/images/integrations/pytorch_example_table.png)

データセットとモデルのロギングと可視化の詳細については、[W&Bテーブルガイド](../data-vis/)をご覧ください。

## PyTorchコードのプロファイリング​

![View detailed traces of PyTorch code execution inside W&B dashboards.](/images/integrations/pytorch_example_dashboard.png)

W&Bは[PyTorch Kineto](https://github.com/pytorch/kineto)の[Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md)と直接統合して、PyTorchコードのプロファイリング用ツールを提供し、CPUとGPU通信の詳細を検査し、ボトルネックと最適化を特定します。


```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # see the profiler docs for details on scheduling
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True)

with profiler:
    ...  # run the code you want to profile here
    # see the profiler docs for detailed usage information

# create a wandb Artifact
profile_art = wandb.Artifact("trace", type="profile")
# add the pt.trace.json files to the Artifact
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# log the artifact
profile_art.save()
```

動作中のコード例を[このColab](http://wandb.me/trace-colab)で参照し、実行してください。

:::caution
インタラクティブなトレース表示ツールはChrome Trace Viewerを基盤とし、Chromeブラウザで最適に動作します。
:::
