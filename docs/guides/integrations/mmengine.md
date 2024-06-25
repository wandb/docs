---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# MMEngine

MMEngineは[OpenMMLab](https://github.com/open-mmlab)によって開発された、PyTorchをベースとしたディープラーニングモデルのトレーニング用基盤ライブラリです。MMEngineはOpenMMLabのアルゴリズムライブラリ向けに次世代のトレーニングアーキテクチャーを実装しており、OpenMMLab内の30以上のアルゴリズムライブラリに対して統一された実行基盤を提供します。その核心コンポーネントには、トレーニングエンジン、評価エンジン、モジュール管理が含まれます。

[Weights and Biases](https://wandb.ai/site)は、専用の[`WandbVisBackend`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.visualization.WandbVisBackend.html#mmengine.visualization.WandbVisBackend)を通じてMMEngineに直接インテグレーションされています。このバックエンドを使用して以下を行うことができます。
- トレーニングおよび評価メトリクスのログ。
- 実験設定のログと管理。
- グラフ、画像、スカラーなどの追加レコードのログ。

## はじめに

まず、`openmim`と`wandb`をインストールする必要があります。その後、`openmim`を使用して`mmengine`と`mmcv`をインストールできます。

<Tabs
  defaultValue="script"
  values={[
    {label: 'Command Line', value: 'script'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install -q -U openmim wandb
mim install -q mmengine mmcv
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install -q -U openmim wandb
!mim install -q mmengine mmcv
```

  </TabItem>
</Tabs>

## `WandbVisBackend`を使用したMMEngine Runnerの利用

このセクションでは、[`mmengine.runner.Runner`](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.runner.Runner.html#mmengine.runner.Runner)を使用して`WandbVisBackend`を利用する典型的なワークフローを示します。

まず、可視化設定から`visualizer`を定義します。

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

# 可視化設定からvisualizerを取得
visualizer = Visualizer.get_instance(**visualization_cfg)
```

:::情報
`init_kwargs`には[W&B run initialization](https://docs.wandb.ai/ref/python/init)の入力パラメータの引数辞書を渡します。
:::

次に、`visualizer`で`runner`を初期化し、`runner.train()`を呼び出します。

```python
from mmengine.runner import Runner

# PyTorchのトレーニングヘルパーであるmmengine Runnerを作成
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict,
    visualizer=visualizer, # visualizerを渡す
)

# トレーニングを開始
runner.train()
```

| ![`WandbVisBackend`を使用してトラッキングされたあなたの実験の一例](@site/static/images/integrations/mmengine.png) | 
|:--:| 
| **`WandbVisBackend`を使用してトラッキングされたあなたの実験の一例。** |

## OpenMMLabコンピュータビジョンライブラリで`WandbVisBackend`を使用

`WandbVisBackend`は、[MMDetection](https://mmdetection.readthedocs.io/)などのOpenMMLabコンピュータビジョンライブラリを使用して実験をトラッキングするのにも簡単に使用できます。

```python
# デフォルトのランタイム設定からベース設定を継承
_base_ = ["../_base_/default_runtime.py"]

# ベース設定の`visualizer`の`vis_backends`に
# `WandbVisBackend`設定辞書を割り当てる
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