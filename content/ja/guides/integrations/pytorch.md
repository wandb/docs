---
title: PyTorch
menu:
  default:
    identifier: ja-guides-integrations-pytorch
    parent: integrations
weight: 300
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

PyTorch は、Python の ディープラーニング において最も人気のある フレームワーク の 1 つで、特に 研究 者の間で人気があります。W&B は、 勾配 の ログ 記録から CPU および GPU での コード のプロファイリングまで、PyTorch を第一級でサポートします。

ぜひ、 Colabノートブック で インテグレーション をお試しください。

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb" >}}

[サンプルリポジトリ](https://github.com/wandb/examples) で スクリプト を確認することもできます。これには、[Fashion MNIST](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) で [Hyperband](https://arxiv.org/abs/1603.06560) を使用した ハイパーパラメーター 最適化に関するものや、それが生成する [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs) などがあります。

## `wandb.watch` で 勾配 を ログ 記録する

勾配 を自動的に ログ 記録するには、[`wandb.watch`]({{< relref path="/ref/python/watch.md" lang="ja" >}}) を呼び出して、PyTorch の モデル を渡します。

```python
import wandb

wandb.init(config=args)

model = ...  # モデル をセットアップ

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

同じ スクリプト で複数の モデル を追跡する必要がある場合は、各 モデル で `wandb.watch` を個別に呼び出すことができます。この関数の リファレンス ドキュメントは [こちら]({{< relref path="/ref/python/watch.md" lang="ja" >}}) にあります。

{{% alert color="secondary" %}}
順方向と逆方向のパスの後に `wandb.log` が呼び出されるまで、 勾配 、 メトリクス 、および グラフ は ログ 記録されません。
{{% /alert %}}

## 画像とメディアを ログ 記録する

画像 データ を含む PyTorch の `Tensors` を [`wandb.Image`]({{< relref path="/ref/python/data-types/image.md" lang="ja" >}}) に渡すことができ、[`torchvision`](https://pytorch.org/vision/stable/index.html) の ユーティリティ が使用されて、自動的に画像に変換されます。

```python
images_t = ...  # PyTorch の Tensor として画像を生成またはロード
wandb.log({"examples": [wandb.Image(im) for im in images_t]})
```

PyTorch およびその他の フレームワーク で リッチメディア を W&B に ログ 記録する方法の詳細については、[メディア ログ 記録 ガイド]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}) を確認してください。

モデル の 予測 や派生した メトリクス など、メディアと一緒に 情報 を含める場合は、`wandb.Table` を使用します。

```python
my_table = wandb.Table()

my_table.add_column("image", images_t)
my_table.add_column("label", labels)
my_table.add_column("class_prediction", predictions_t)

# W&B に テーブル を ログ 記録
wandb.log({"mnist_predictions": my_table})
```

{{< img src="/images/integrations/pytorch_example_table.png" alt="The code above generates a table like this one. This model's looking good!" >}}

データセット と モデル の ログ 記録と視覚化の詳細については、[W&B Tables の ガイド]({{< relref path="/guides/models/tables/" lang="ja" >}}) を確認してください。

## PyTorch コード のプロファイル

{{< img src="/images/integrations/pytorch_example_dashboard.png" alt="View detailed traces of PyTorch code execution inside W&B dashboards." >}}

W&B は、[PyTorch Kineto](https://github.com/pytorch/kineto) の [Tensorboard plugin](https://github.com/pytorch/kineto/blob/master/tb_plugin/README.md) と直接 統合 されており、PyTorch コード のプロファイル、CPU と GPU の通信の詳細の検査、および ボトルネック と 最適化 の特定のための ツール を提供します。

```python
profile_dir = "path/to/run/tbprofile/"
profiler = torch.profiler.profile(
    schedule=schedule,  # スケジューリング の詳細については、プロファイラー のドキュメントを参照してください
    on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    with_stack=True,
)

with profiler:
    ...  # ここでプロファイルするコードを実行
    # 詳細な使用方法については、プロファイラー のドキュメントを参照してください

# wandb Artifact を作成
profile_art = wandb.Artifact("trace", type="profile")
# pt.trace.json ファイルを Artifact に追加
profile_art.add_file(glob.glob(profile_dir + ".pt.trace.json"))
# artifact をログに記録
profile_art.save()
```

動作する サンプル コード を [この Colab](http://wandb.me/trace-colab) で確認して実行してください。

{{% alert color="secondary" %}}
インタラクティブな トレース 表示 ツール は Chrome Trace Viewer に基づいており、Chrome ブラウザー で最適に動作します。
{{% /alert %}}
