---
title: PyTorch
menu:
  default:
    identifier: pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、特に研究者の間で人気の高い Python 用ディープラーニングフレームワークのひとつです。W&B は PyTorch を強力にサポートしており、勾配のログから CPU・GPU 上でのコードのプロファイリングまで対応しています。

Colabノートブックでインテグレーションを試してみてください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

各種スクリプトの例については、[example repo](https://github.com/wandb/examples) をご覧ください。ここには、[Hyperband](https://arxiv.org/abs/1603.06560) を使ったハイパーパラメーター最適化や [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) の例、さらには生成される [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) も掲載しています。

## `run.watch` で勾配をログする

勾配を自動でログしたい場合は、[`wandb.Run.watch()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runwatch" >}}) を呼び出して、PyTorch モデルを渡すことができます。

```python
import wandb

with wandb.init(config=args) as run:

    model = ...  # モデルをセットアップ

    # マジック
    run.watch(model, log_freq=100)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            run.log({"loss": loss})
```

同一スクリプトで複数のモデルをトラッキングしたい場合は、それぞれのモデルに対して [`wandb.Run.watch()`]({{< relref "/ref/python/sdk/classes/run/#method-runwatch" >}}) を個別に呼んでください。

{{% alert color="secondary" %}}
勾配・メトリクス・グラフは、前方 _かつ_ 後方パスの後で `wandb.Run.log()` を呼び出すまでログされません。
{{% /alert %}}

## 画像やメディアのログ

PyTorch の画像データを持つ `Tensor` を [`wandb.Image`]({{< relref "/ref/python/sdk/data-types/image.md" >}}) に渡すと、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティが自動で画像に変換してくれます。

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    images_t = ...  # PyTorch Tensor として画像を生成または読み込み
    run.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch やその他のフレームワークでリッチメディアをログする詳細については、[media logging guide]({{< relref "/guides/models/track/log/media.md" >}}) もご覧ください。

メディアと同時に、モデルの予測や算出したメトリクスなどの情報も含めたい場合は、`wandb.Table` を使いましょう。

```python
with wandb.init() as run:
    my_table = wandb.Table()

    my_table.add_column("image", images_t)
    my_table.add_column("label", labels)
    my_table.add_column("class_prediction", predictions_t)

    # Table を W&B にログ
    run.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="PyTorch model results" >}}

データセットやモデルのログ・可視化については、[W&B Tables のガイド]({{< relref "/guides/models/tables/" >}}) もぜひご参照ください。

## PyTorch コードのプロファイリング

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="PyTorch execution traces" >}}

W&B は [PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard プラグイン](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接連携し、PyTorch コードのプロファイリング、CPU と GPU 間の通信詳細の確認、ボトルネックや最適化ポイントの特定などのツールを提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細は profiler ドキュメント参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # プロファイリングしたいコードをここで実行
    # 詳細な使い方は profiler ドキュメントを参照

# wandb Artifact を作成
profile_art = wandb.Artifact("trace", type="profile")
# Artifact に pt.trace.json ファイルを追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# Artifact を保存してログ
profile_art.save()
```

動作するサンプルコードは [こちらの Colab](https://wandb.me/trace-colab) からご確認いただけます。

{{% alert color="secondary" %}}
インタラクティブなトレースビューツールは Chrome Trace Viewer をベースにしており、Chrome ブラウザでの利用を推奨しています。
{{% /alert %}}