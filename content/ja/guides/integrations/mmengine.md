---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

MMEngine は [OpenMMLab](https://github.com/open-mmlab) によって開発された、PyTorch をベースとした ディープラーニング モデルのトレーニングを行うための基盤 ライブラリです。MMEngine は OpenMMLab アルゴリズム ライブラリの次世代トレーニング アーキテクチャーを実装し、OpenMMLab 内の 30 以上のアルゴリズム ライブラリに統一された実行基盤を提供します。そのコア コンポーネントには、トレーニング エンジン、評価エンジン、およびモジュール管理が含まれます。

[Weights and Biases](https://wandb.ai/site) は、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を介して MMEngine に直接統合されており、以下のことが可能です。
- トレーニング および 評価 メトリクスを ログに記録する。
- 実験 の config を ログに記録および管理する。
- グラフ、画像、スカラーなどの追加レコードを ログに記録する。

## はじめに

`openmim` と `wandb` をインストールします。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="script" %}}

``` bash
pip install -q -U openmim wandb
```

{{% /tab %}}

{{% tab header="ノートブック" value="notebook" %}}

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

{{% tab header="ノートブック" value="notebook" %}}

``` bash
!mim install -q mmengine mmcv
```

{{% /tab %}}
{{< /tabpane >}}

## `WandbVisBackend` を MMEngine Runner で使用する

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を使用して `WandbVisBackend` を使用する典型的な ワークフロー を示します。

1. 可視化 config から `visualizer` を定義します。

    ```python
    from mmengine.visualization import Visualizer

    # 可視化 config を定義します
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

    # 可視化 config から visualizer を取得します
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B run 初期化]({{< relref path="/ref/python/init" lang="ja" >}}) の入力 パラメータ に、 引数 の 辞書 を `init_kwargs` に渡します。
    {{% /alert %}}

2. `visualizer` で `runner` を初期化し、`runner.train()` を呼び出します。

    ```python
    from mmengine.runner import Runner

    # PyTorch のトレーニング ヘルパーである mmengine Runner を構築します
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # visualizer を渡します
    )

    # トレーニングを開始します
    runner.train()
    ```

## OpenMMLab コンピュータビジョン ライブラリで `WandbVisBackend` を使用する

`WandbVisBackend` は、[MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab コンピュータビジョン ライブラリ で 実験 を追跡するためにも簡単に使用できます。

```python
# デフォルトの ランタイム config からベース config を継承します
_base_ = ["../_base_/default_runtime.py"]

# `WandbVisBackend` config 辞書を、
# ベース config からの `visualizer` の `vis_backends` に割り当てます
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