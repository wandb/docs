---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、Python における ディープラーニング で最も人気のある フレームワーク の 1 つであり、特に 研究 者の間で人気があります。W&B は、 勾配 の ログ 記録から CPU および GPU での コード のプロファイリングまで、PyTorch をファーストクラスでサポートしています。

ぜひ、 Colabノートブック で インテグレーション をお試しください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[サンプル repo](https://github.com/wandb/examples) には、[Hyperband](https://arxiv.org/abs/1603.06560) を使用した ハイパーパラメーター 最適化に関するもの ([Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) 上) や、それが生成する [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) など、 スクリプト がありますので、こちらもご覧ください。

## `wandb.watch` で 勾配 を ログ 記録する

自動的に 勾配 を ログ 記録するには、[`wandb.watch`]({{< relref path="/ref/python/watch.md" lang="ja" >}}) を呼び出し、PyTorch の model を渡します。

```python
import wandb

wandb.init(config=args)

model = ...  # model をセットアップする

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

同じ スクリプト で複数の model を追跡する必要がある場合は、各 model に対して個別に `wandb.watch` を呼び出すことができます。この関数のリファレンスドキュメントは[こちら]({{< relref path="/ref/python/watch.md" lang="ja" >}})にあります。

{{% alert color="secondary" %}}
順伝播 _および_ 逆伝播 の後に `wandb.log` が呼び出されるまで、 勾配 、 メトリクス 、および グラフ は ログ 記録されません。
{{% /alert %}}

## 画像とメディアの ログ 記録

PyTorch の `Tensor` を画像 データ とともに [`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}}) に渡すことができ、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティを使用して、それらを自動的に画像に変換できます。

```python
images_t = ...  # PyTorch Tensor として画像を生成またはロードする
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch およびその他の フレームワーク で リッチメディア を W&B に ログ 記録する方法の詳細については、[メディア ログ 記録 ガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}})をご覧ください。

model の 予測 や 派生 メトリクス など、メディアとともに 情報 も含めたい場合は、`wandb.Table` を使用します。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&B に Table を ログ 記録する
wandb.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="The code above generates a table like this one. This model's looking good!" >}}

データセット と model の ログ 記録と視覚化の詳細については、[W&B Tables の ガイド]({{< relref path="/guides/core/tables/" lang="ja" >}})をご覧ください。

## PyTorch コード のプロファイル

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="View detailed traces of PyTorch code execution inside W&B dashboards." >}}

W&B は、[PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接 統合 されており、PyTorch コード のプロファイル、CPU および GPU 通信の詳細の検査、ボトルネックと最適化の特定を行う ツール を提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細については、プロファイラーのドキュメントを参照してください
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # ここでプロファイルする コード を実行します
    # 詳細な使用方法については、プロファイラーのドキュメントを参照してください

# wandb Artifact を作成する
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルを Artifact に追加する
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# artifact を ログ 記録する
profile_art.save()
```

[この Colab](http://wandb.me/trace-colab) で 動作 する サンプルコード を確認して実行してください。

{{% alert color="secondary" %}}
インタラクティブな トレース 表示 ツール は Chrome Trace Viewer に基づいており、Chrome ブラウザーで最適に動作します。
{{% /alert %}}
