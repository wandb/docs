---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

OpenMMLab の MMEngine ( [OpenMMLab](https://github.com/open-mmlab) ) は、PyTorch をベースにしたディープラーニング モデルをトレーニングするための基盤ライブラリです。MMEngine は、OpenMMLab アルゴリズム ライブラリの次世代トレーニング アーキテクチャーを実装し、OpenMMLab 内の 30 以上のアルゴリズム ライブラリに統一された実行基盤を提供します。そのコア コンポーネントには、トレーニング エンジン、評価エンジン、およびモジュール管理が含まれます。

[Weights and Biases](https://wandb.ai/site) は、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を通じて MMEngine に直接統合されており、次の用途に使用できます。
- トレーニング および 評価 メトリクス の ログ記録。
- 実験 configs の ログ記録と管理。
- グラフ、画像、スカラーなどの追加レコードのログ記録。

## はじめに

`openmim` と `wandb` をインストールします。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="script" %}}

``` bash
pip install -q -U openmim wandb
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

``` bash
!pip install -q -U openmim wandb
```

{{% /tab %}}
{{< /tabpane >}}

次に、`mim` を使用して `mmengine` と `mmcv` をインストールします。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="script" %}}

``` bash
mim install -q mmengine mmcv
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

``` bash
!mim install -q mmengine mmcv
```

{{% /tab %}}
{{< /tabpane >}}

## MMEngine Runner で `WandbVisBackend` を使用する

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を使用した `WandbVisBackend` を使用した典型的な ワークフロー を示します。

1. 可視化 config から `visualizer` を定義します。

    ```python
    from mmengine.visualization import Visualizer

    # define the visualization configs
    visualization_cfg = dict(
        name="wandb_visualizer",
        vis_backends=[
            dict(
                type='WandbVisBackend',
                init_kwargs=dict(project="mmengine"),
            )
        ],
        save_dir="runs/wandb"
    )

    # get the visualizer from the visualization configs
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B の run 初期化]({{< relref path="/ref/python/init" lang="ja" >}}) の入力 パラメータ の 引数 の 辞書 を `init_kwargs` に渡します。
    {{% /alert %}}

2. `visualizer` で `runner` を初期化し、`runner.train()` を呼び出します。

    ```python
    from mmengine.runner import Runner

    # build the mmengine Runner which is a training helper for PyTorch
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # pass the visualizer
    )

    # start training
    runner.train()
    ```

## OpenMMLab コンピュータビジョン ライブラリで `WandbVisBackend` を使用する

`WandbVisBackend` を使用すると、[MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab コンピュータビジョン ライブラリを使用した 実験 の追跡も簡単に行えます。

```python
# inherit base configs from the default runtime configs
_base_ = ["../_base_/default_runtime.py"]

# Assign the `WandbVisBackend` config dictionary to the
# `vis_backends` of the `visualizer` from the base configs
_base_.visualizer.vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs={
            'project': 'mmdet',
            'entity': 'geekyrakshit'
        },
    ),
]
```