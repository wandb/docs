---
title: MMEngine
menu:
  default:
    identifier: ja-guides-integrations-mmengine
    parent: integrations
weight: 210
---

MMEngine は [OpenMMLab](https://github.com/open-mmlab) による PyTorch ベースのディープラーニングモデルのトレーニング用基盤ライブラリです。MMEngine は OpenMMLab アルゴリズムライブラリ向けの次世代トレーニングアーキテクチャーを実装しており、OpenMMLab 内の30以上のアルゴリズムライブラリに対し統一的な実行基盤を提供します。主なコアコンポーネントは、トレーニングエンジン、評価エンジン、そしてモジュール管理です。

[W&B](https://wandb.ai/site) は、専用の [`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend) を通じて MMEngine と直接統合されており、次のような用途に利用できます。
- トレーニングおよび評価メトリクスのログ
- 実験設定(config)のログおよび管理
- グラフ・画像・スカラー値などの追加データのログ

## はじめに

まずは `openmim` と `wandb` をインストールします。

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

次に、`mim` を使って `mmengine` および `mmcv` をインストールします。

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

## MMEngine Runner で `WandbVisBackend` を利用する

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner) を使った典型的なワークフローでの `WandbVisBackend` の利用例を紹介します。

1. visualization config から `visualizer` を定義します。

    ```python
    from mmengine.visualization import Visualizer

    # visualization 設定を定義
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

    # visualization config から visualizer インスタンスを取得
    visualizer = Visualizer.get_instance(**visualization_cfg)
    ```

    {{% alert %}}
    `init_kwargs` には [W&B run 初期化]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の入力パラメータ用引数の辞書を渡すことができます。
    {{% /alert %}}

2. `visualizer` を指定して `runner` を初期化し、`runner.train()` を呼びます。

    ```python
    from mmengine.runner import Runner

    # mmengine Runner を構築（PyTorch 用トレーニング支援ツール）
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

## OpenMMLab のコンピュータビジョンライブラリで `WandbVisBackend` を使う

`WandbVisBackend` は [MMDetection](https://mmdetection.readthedocs.io/) などの OpenMMLab のコンピュータビジョン系ライブラリで実験管理を行う際にも簡単に利用可能です。

```python
# デフォルトのランタイム config から base config を継承
_base_ = ["../_base_/default_runtime.py"]

# base config の visualizer の `vis_backends` に
# `WandbVisBackend` の設定辞書を割り当て
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