---
displayed_sidebar: default
---


# PyTorch

[**Try in a Colab Notebook here →**](http://wandb.me/intro)

PyTorchは、特に研究者の間で、Pythonで最も人気のあるディープラーニングフレームワークの一つです。W&Bは、勾配のログ記録からCPUやGPUでのコードのプロファイリングまで、PyTorchに対する一級のサポートを提供します。

:::info
[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple\_PyTorch\_Integration.ipynb)（以下の動画ガイド付き）でインテグレーションを試すか、[example repo](https://github.com/wandb/examples)でスクリプトを確認してください。スクリプトには、[Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)で[Hyperband](https://arxiv.org/abs/1603.06560)を使用してハイパーパラメーターを最適化する方法も含まれています。また、生成される[W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)もご覧ください。
:::

## `wandb.watch`を使った勾配のログ記録

勾配を自動的にログに記録するために、[`wandb.watch`](../../ref/python/watch.md)を呼び出し、PyTorchモデルを渡すことができます。

```python
import wandb

wandb.init(config=args)

model = ...  # モデルを設定

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

同じスクリプトで複数のモデルを追跡する必要がある場合は、各モデルに対して `wandb.watch` を個別に呼び出します。この関数のリファレンスドキュメントは[こちら](../../ref/python/watch.md)です。

:::caution
勾配、メトリクス、およびグラフは、フォワードパスとバックワードパスの後に `wandb.log` が呼び出されるまでログに記録されません。
:::

## イメージやメディアをログに記録

PyTorchの`Tensors`をイメージデータとして[`wandb.Image`](../../ref/python/data-types/image.md)に渡すと、[`torchvision`](https://pytorch.org/vision/stable/index.html)のユーティリティを使用して自動的に画像に変換されます。

```python
images_t = ...  # PyTorchテンソルとして画像を生成またはロード
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorchや他のフレームワークでのリッチメディアのログに関する詳細は、[メディアログガイド](../track/log/media.md)をご覧ください。

また、メディアと一緒にモデルの予測や派生メトリクスなどの情報を含めたい場合は、`wandb.Table`を使用します。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# TableをW&Bにログ
wandb.log({"mnist_predictions": my_table})
```

![上記のコードはこのようなテーブルを生成します。このモデルは良好です！](/images/integrations/pytorch_example_table.png)

データセットとモデルのログおよび視覚化に関する詳細は、[W&B Tablesのガイド](../tables/intro.md)をご覧ください。

## PyTorchコードのプロファイリング

![W&Bダッシュボード内でPyTorchコードの実行の詳細なトレースを表示します。](/images/integrations/pytorch_example_dashboard.png)

W&Bは、[PyTorch Kineto](https://github.com/pytorch/kineto)の[Tensorboardプラグイン](https://github.com/pytorch/kineto/blob/master/tb\_plugin/README.md)と直接統合されており、PyTorchコードのプロファイリングツール、CPUとGPUの通信の詳細を検査し、ボトルネックや最適化を特定するツールを提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細はプロファイラのドキュメントを参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # ここにプロファイルしたいコードを実行
    # 詳細な使用情報についてはプロファイラのドキュメントを参照

# wandb Artifactを作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.jsonファイルをArtifactに追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# アーティファクトを保存
profile_art.save()
```

[このColab](http://wandb.me/trace-colab)で動作するサンプルコードを確認し、実行してください。

:::caution
インタラクティブなトレース表示ツールはChrome Trace Viewerを基にしており、Chromeブラウザで最適に動作します。
:::
