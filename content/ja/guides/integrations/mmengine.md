---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

[OpenMMLab](https://github.com/open-mmlab) の MMEngine は、PyTorch ベースの ディープラーニング モデルをトレーニング するための基盤 ライブラリです。MMEngine は OpenMMLab のアルゴリズム ライブラリ向けに次世代のトレーニング アーキテクチャーを実装しており、OpenMMLab 内の 30 を超えるアルゴリズム ライブラリに対して統一された実行基盤を提供します。中核コンポーネントには、トレーニングエンジン、評価エンジン、モジュール管理が含まれます。

[W&B](https://wandb.ai/site) は、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を通じて MMEngine に直接統合されており、次の用途に使えます。
- トレーニング と 評価 のメトリクスをログに記録する。
- 実験 の設定をログ・管理する。
- グラフ、画像、スカラー などの追加の記録をログに残す。

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

次に、`mim` を使って `mmengine` と `mmcv` をインストールします。

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

## MMEngine Runner で `WandbVisBackend` を使う

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を使って `WandbVisBackend` を用いる一般的なワークフローを示します。

1. 可視化設定から `visualizer` を定義します。

    ```python
    from mmengine.visualization import Visualizer

    # 可視化設定を定義
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

    # 可視化設定から visualizer を取得
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    [W&B run の初期化]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の入力パラメータとなる引数の辞書を `init_kwargs` に渡します。
    {{% /alert %}}

2. `visualizer` を渡して `runner` を初期化し、`runner.train()` を呼び出します。

    ```python
    from mmengine.runner import Runner

    # PyTorch 用のトレーニング支援である mmengine Runner を構築
    runner = Runner(
        model,
        work_dir='runs/gan/',
        train_dataloader=train_dataloader,
        train_cfg=train_cfg,
        optim_wrapper=opt_wrapper_dict,
        visualizer=visualizer, # visualizer を渡す
    )

    # トレーニング開始
    runner.train()
    ```

## OpenMMLab の コンピュータビジョン ライブラリで `WandbVisBackend` を使う

`WandbVisBackend` は、[MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab の コンピュータビジョン ライブラリで 実験 を追跡する用途にも簡単に使えます。

```python
# 既定のランタイム設定からベース設定を継承
_base_ = ["../_base_/default_runtime.py"]

# `WandbVisBackend` の設定辞書を
# ベース設定の `visualizer` の `vis_backends` に割り当て
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