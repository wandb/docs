---
title: 'MMEngine

  '
menu:
  default:
    identifier: mmengine
    parent: integrations
weight: 210
---

MMEngine は [OpenMMLab](https://github.com/open-mmlab) によって開発された PyTorch ベースのディープラーニングモデルのトレーニング用基盤ライブラリです。MMEngine は OpenMMLab アルゴリズムライブラリ向けの次世代トレーニングアーキテクチャーを実装しており、OpenMMLab 内の 30 を超えるアルゴリズムライブラリに対して統一された実行基盤を提供します。主なコンポーネントにはトレーニングエンジン、評価エンジン、モジュール管理などがあります。

[W&B](https://wandb.ai/site) は MMEngine に直接統合されており、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を利用できます。これにより、
- トレーニングや評価のメトリクスをログできます。
- 実験構成（config）を記録・管理できます。
- グラフや画像、スカラー値などの追加情報もログできます。

## はじめに

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

次に、`mim` を使って `mmengine` と `mmcv` をインストールします。

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

## MMEngine Runner で `WandbVisBackend` を使う

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を用いた典型的なワークフローで `WandbVisBackend` を利用する方法を紹介します。

1. 可視化 config を使って `visualizer` を定義します。

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

    # 設定から visualizer を取得
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B run 初期化]({{< relref "/ref/python/sdk/functions/init.md" >}}) の入力パラメータ（引数）の辞書を `init_kwargs` に渡すことができます。
    {{% /alert %}}

2. `visualizer` を指定して `runner` を初期化し、`runner.train()` を呼びます。

    ```python
    from mmengine.runner import Runner

    # mmengine Runner（PyTorch 用のトレーニングヘルパー）を構築
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # visualizer を指定
    )

    # トレーニング開始
    runner.train()
    ```

## OpenMMLab コンピュータビジョンライブラリで `WandbVisBackend` を利用する

`WandbVisBackend` は [MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab コンピュータビジョンライブラリにおける experiment のトラッキングにも手軽に使えます。

```python
# ベースの runtime 設定ファイルから基本設定を継承
_base_ = ["../_base_/default_runtime.py"]

# ベース設定の visualizer の vis_backends に
# `WandbVisBackend` 用の設定辞書を割り当て
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