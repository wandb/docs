---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}
PyTorch は、特に研究者の間で人気の高い、Python 向けディープラーニング用フレームワークのひとつです。W&B は PyTorch を強力にサポートしており、勾配のログから CPU や GPU 上でのコードのプロファイリングまで対応しています。
Colab ノートブックでこのインテグレーションを試してみてください。
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}
スクリプトは [サンプル リポジトリ](https://github.com/wandb/examples) でも確認できます。たとえば [Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) に対する [Hyperband](https://arxiv.org/abs/1603.06560) を用いたハイパーパラメーター最適化の例や、そこから生成される [W&B ダッシュボード](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) があります。

## `run.watch` で勾配をログする

勾配を自動でログするには、[`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runwatch" lang="ja" >}}) を呼び出し、PyTorch のモデルを渡します。

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

同じスクリプト内で複数のモデルを追跡する必要がある場合は、各モデルに対して個別に [`wandb.Run.watch()`]({{< relref path="/ref/python/sdk/classes/run/#method-runwatch" lang="ja" >}}) を呼び出してください。

{{% alert color="secondary" %}}
`wandb.Run.log()` が順伝播 _と_ 逆伝播のあとに呼び出されるまで、勾配、メトリクス、計算グラフはログされません。
{{% /alert %}}

## 画像やメディアをログする

画像データを持つ PyTorch の `Tensors` を [`wandb.Image`]({{< relref path="/ref/python/sdk/data-types/image.md" lang="ja" >}}) に渡すと、[`torchvision`](https://pytorch.org/vision/stable/index.html) のユーティリティが自動的に画像へ変換します。

```python
with wandb.init(project="my_project", entity="my_entity") as run:
    images_t = ...  # PyTorch の Tensor として画像を生成または読み込み
    run.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch やその他のフレームワークでリッチメディアを W&B にログする詳細は、[メディア ログ ガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) を参照してください。

メディアと一緒に、モデルの予測や派生メトリクスなどの情報も含めたい場合は、`wandb.Table` を使ってください。

```python
with wandb.init() as run:
    my_table = wandb.Table()

    my_table.add_column("image", images_t)
    my_table.add_column("label", labels)
    my_table.add_column("class_prediction", predictions_t)

    # Table を W&B にログ
    run.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="PyTorch モデルの結果" >}}

データセットやモデルのログと可視化の詳細は、[W&B Tables のガイド]({{< relref path="/guides/models/tables/" lang="ja" >}}) を参照してください。

## PyTorch コードをプロファイルする

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="PyTorch 実行トレース" >}}

W&B は [PyTorch Kineto](https://github.com/pytorch/kineto) の [TensorBoard プラグイン](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接連携し、PyTorch コードのプロファイリング、CPU と GPU 間通信の詳細の確認、ボトルネックや最適化ポイントの特定を行うツールを提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリングの詳細はプロファイラのドキュメントを参照
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # プロファイルしたいコードをここで実行
    # 詳細な使い方はプロファイラのドキュメントを参照

# wandb Artifact を作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルを Artifact に追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# Artifact をログ
profile_art.save()
```

実行可能なサンプルコードは [この Colab](https://wandb.me/trace-colab) で確認・実行できます。

{{% alert color="secondary" %}}
インタラクティブなトレース閲覧ツールは Chrome Trace Viewer に基づいており、Chrome ブラウザでの利用が最適です。
{{% /alert %}}