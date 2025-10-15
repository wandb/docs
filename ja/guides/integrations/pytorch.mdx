---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、特に研究者の間で、Python におけるディープラーニングの最も人気のあるフレームワークの一つです。W&B は、PyTorch に対して一流のサポートを提供し、勾配のログから CPU と GPU 上でのコードのプロファイリングまで対応しています。

Colab ノートブックで私たちのインテグレーションを試してみてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

また、[example repo](https://github.com/wandb/examples) では、スクリプトや [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) を使用した [Hyperband](https://arxiv.org/abs/1603.06560) によるハイパーパラメータ最適化などの例を含むものがあります。それが生成する [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) もご覧いただけます。

## `wandb.watch` を使った勾配のログ

勾配を自動的にログするには、[`wandb.watch`]({{< relref path="/ref/python/watch.md" lang="ja" >}}) を呼び出して、PyTorch モデルを渡します。

```python
import wandb

wandb.init(config=args)

model = ...  # モデルをセットアップする

# マジック
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

同じスクリプト内で複数のモデルを追跡する必要がある場合は、それぞれのモデルに対して `wandb.watch` を個別に呼び出すことができます。この関数の参照ドキュメントは[こちら]({{< relref path="/ref/python/watch.md" lang="ja" >}})。

{{% alert color="secondary" %}}
勾配、メトリクス、およびグラフは、フォワード _および_ バックワードパスの後に `wandb.log` が呼び出されるまでログされません。
{{% /alert %}}

## 画像とメディアのログ

画像データを持つ PyTorch `Tensors` を [`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}}) に渡すことができ、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティが自動的に画像に変換します。

```python
images_t = ...  # PyTorch Tensors として画像を生成またはロードする
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch や他のフレームワークにおけるリッチメディアのログについての詳細は、[メディアログガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})をご覧ください。

メディアと一緒にモデルの予測や派生メトリクスなどの情報も含めたい場合は、`wandb.Table` を使用します。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# Table を W&B にログ
wandb.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="上記のコードはこのようなテーブルを生成します。このモデルは良好に見えます！" >}}

データセットやモデルのログと視覚化についての詳細は、[W&B Tables のガイド]({{< relref path="/guides/models/tables/" lang="ja" >}})をご覧ください。

## PyTorch コードのプロファイリング

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="W&B ダッシュボード内で PyTorch コード実行の詳細なトレースを確認します。" >}}

W&B は [PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard プラグイン](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接統合されており、PyTorch コードのプロファイリング、CPU と GPU の通信の詳細の検査、ボトルネックや最適化を識別するためのツールを提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジュールの詳細はプロファイラードキュメントを参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # プロファイルしたいコードをここで実行
    # 詳細な使用情報はプロファイラードキュメントを参照

# wandb アーティファクトを作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルをアーティファクトに追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# アーティファクトをログ
profile_art.save()
```

[こちらの Colab](http://wandb.me/trace-colab)で作業中の例コードを見て実行できます。

{{% alert color="secondary" %}}
インタラクティブなトレースビューツールは、Chrome Trace Viewer に基づいており、Chrome ブラウザで最も良好に動作します。
{{% /alert %}}