---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

MMEngine by [OpenMMLab](https://github.com/open-mmlab) は、PyTorch に基づくディープラーニングモデルのトレーニング用の基盤ライブラリです。MMEngine は、OpenMMLab アルゴリズムライブラリの次世代トレーニングアーキテクチャを実装しており、OpenMMLab 内の 30 を超えるアルゴリズムライブラリに対して統一された実行基盤を提供します。そのコアコンポーネントには、トレーニングエンジン、評価エンジン、およびモジュール管理が含まれます。

[Weights and Biases](https://wandb.ai/site) は、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を通じて MMEngine に直接統合されており、以下のことができます。
- トレーニングおよび評価メトリクスをログする。
- 実験の設定をログして管理する。
- グラフ、画像、スカラーなどの追加の記録をログする。

## 始めましょう

`openmim` と `wandb` をインストールします。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="script" %}}

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
{{% tab header="Command Line" value="script" %}}

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

## `WandbVisBackend` を MMEngine Runner で使用する

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を使用して `WandbVisBackend` を用いた典型的なワークフローを示します。

1. 可視化設定から `visualizer` を定義する。

    ```python
    from mmengine.visualization import Visualizer

    # 可視化設定を定義する
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

    # 可視化設定から visualizer を取得する
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B run initialization]({{< relref path="/ref/python/init" lang="ja" >}}) 入力パラメータに対して、`init_kwargs` に引数の辞書を渡します。
    {{% /alert %}}

2. `visualizer` で `runner` を初期化し、`runner.train()` を呼び出す。

    ```python
    from mmengine.runner import Runner

    # PyTorch のトレーニングヘルパーである mmengine Runner を構築する
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # visualizer を渡す
    )

    # トレーニングを開始する
    runner.train()
    ```

## OpenMMLab コンピュータビジョンライブラリで `WandbVisBackend` を使用する

`WandbVisBackend` は、[MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab コンピュータビジョンライブラリを使用して実験管理を行う際にも簡単に利用できます。

```python
# デフォルトのランタイム設定からベース設定を継承する
_base_ = ["../_base_/default_runtime.py"]

# ベース設定から `visualizer` の `vis_backends` に `WandbVisBackend` の設定辞書を割り当てる
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