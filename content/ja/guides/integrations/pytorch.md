---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、Python で特に研究者に人気のあるディープラーニング用フレームワークの一つです。W&B は PyTorch を強力にサポートしており、勾配のログから、CPU や GPU 上でのコードのプロファイリングまで対応しています。

ぜひ Colabノートブック でインテグレーションをお試しください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

また、[example repo](https://github.com/wandb/examples) ではさまざまなスクリプトを公開しており、[Hyperband](https://arxiv.org/abs/1603.06560) を使ったハイパーパラメーター最適化や [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) の例、さらに生成された [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) もご覧いただけます。

## `run.watch` で勾配をログする

自動的に勾配をログするには、[`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runwatch" lang="ja" >}}) を呼び出して、PyTorch モデルを渡します。

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

同じスクリプト内で複数のモデルを追跡する必要がある場合は、各モデルごとに [`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run/#method-runwatch" lang="ja" >}}) を個別に呼び出してください。

{{% alert color="secondary" %}}
勾配、メトリクス、グラフがログされるのは、フォワードパス _および_ バックワードパスの後に `wandb.Run.log()` が呼ばれたタイミングです。
{{% /alert %}}

## 画像やメディアをログする

PyTorch の画像 `Tensor` は、そのまま [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) に渡すことができ、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティで自動的に画像へ変換されます。

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    images_t = ...  # PyTorch Tensor で画像を生成または読み込む
    run.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch や他のフレームワークで W&B へリッチなメディアを記録する方法については、[メディアログガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) をご覧ください。

また、メディアの横にモデルの予測やメトリクスなどの情報を含めたい場合は、`wandb.Table` を使いましょう。

```python
with wandb.init() as run:
    my_table = wandb.Table()

    my_table.add_column("image", images_t)
    my_table.add_column("label", labels)
    my_table.add_column("class_prediction", predictions_t)

    # Table を W&B へログ
    run.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="PyTorch モデルの結果" >}}

データセットやモデルのログ・可視化については、[W&B Tables のガイド]({{< relref path="/guides/models/tables/" lang="ja" >}}) もご参照ください。

## PyTorch コードをプロファイリングする

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="PyTorch の実行トレース" >}}

W&B は [PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard プラグイン](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接連携し、PyTorch コードのプロファイリングや、CPU/GPU間の通信の詳細分析、ボトルネックの特定・最適化をサポートします。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細はプロファイラのドキュメントを参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # プロファイリングしたいコードをここで実行
    # 詳細な使い方はプロファイラのドキュメントを参照

# wandb Artifact を作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルを Artifact に追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# アーティファクトをログ
profile_art.save()
```

[この Colab](https://wandb.me/trace-colab) で、実際に動作するコード例と実行結果を体験できます。

{{% alert color="secondary" %}}
インタラクティブなトレースビューイング ツールは Chrome Trace Viewer をベースにしており、Chrome ブラウザでの利用が最適です。
{{% /alert %}}