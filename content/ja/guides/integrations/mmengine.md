---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

MMEngine by [OpenMMLab](https://github.com/open-mmlab) は、PyTorch に基づくディープラーニングモデルのトレーニングのための基盤ライブラリです。MMEngine は OpenMMLab のアルゴリズムライブラリ用の次世代のトレーニングアーキテクチャーを実装し、OpenMMLab 内の30以上のアルゴリズムライブラリに対して統一された実行基盤を提供します。そのコアコンポーネントには、トレーニングエンジン、評価エンジン、モジュール管理が含まれます。

[Weights and Biases](https://wandb.ai/site) は、専用の[`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend)を通じて MMEngine に直接統合されています。これを使用して
- トレーニングおよび評価メトリクスをログする。
- 実験設定をログおよび管理する。
- グラフ、画像、スカラーなどの追加記録をログする。

## はじめに

`openmim` および `wandb` をインストールします。

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

次に、`mim` を使用して `mmengine` および `mmcv` をインストールします。

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

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner)を使用した `WandbVisBackend` の典型的なワークフローを示します。

1. 可視化設定から `visualizer` を定義します。

    ```python
    from mmengine.visualization import Visualizer

    # 可視化の設定を定義する
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
    [W&B run 初期化]({{< relref path="/ref/python/init" lang="ja" >}})の入力パラメータ用引数の辞書を `init_kwargs` に渡します。
    {{% /alert %}}

2. `visualizer` とともに `runner` を初期化し、`runner.train()` を呼び出します。

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

## `WandbVisBackend` を OpenMMLab コンピュータビジョンライブラリで使用する

`WandbVisBackend` は、[MMDetection](https://mmdetection.readthedocs.io/) のような OpenMMLab コンピュータビジョンライブラリを使って実験管理を追跡するためにも簡単に使用できます。

```python
# デフォルトのランタイム設定から基本設定を継承する
_base_ = ["../_base_/default_runtime.py"]

# base configs から `visualizer` の `vis_backends` に
# `WandbVisBackend` の設定辞書を割り当てる
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