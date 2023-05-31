---
slug: /guides/integrations/deepchem
description: How to integrate W&B with DeepChem library.
displayed_sidebar: ja
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# DeepChem

[DeepChemライブラリ](https://github.com/deepchem/deepchem)は、薬物発見、材料科学、化学、生物学においてディープラーニングの使用を民主化するオープンソースツールを提供しています。このW&Bとの統合により、DeepChemを使用してモデルをトレーニングする際に簡単で使いやすい実験トラッキングとモデルチェックポイントを追加できます。

## 🧪 DeepChemエクスペリメントの3行のコードでのログ記録

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

![](@site/static/images/integrations/cd.png)

## レポート＆Google Colab

W&B DeepChem統合を使用して生成されたチャートの例として、[W&B with DeepChem：分子グラフ畳み込みネットワーク](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) の記事をご覧ください。

すぐに作業コードに飛び込む場合は、[**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb)をチェックしてください。

## はじめに: 実験のトラッキング

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) のタイプのDeepChemモデル用にWeights & Biasesをセットアップします。
### 1) `wandb`ライブラリをインストールしてログインする

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) WandbLoggerを初期化して設定する
```python
from deepchem.models import WandbLogger

logger = WandbLogger(entity="my_entity", project="my_project")
```

### 3) トレーニングと評価データをW&Bにログする

トレーニングロスと評価メトリクスは、自動的にWeights & Biasesにログされます。オプションの評価は、DeepChemの[ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py)を使用して有効にすることができます。`WandbLogger`は、ValidationCallbackコールバックを検出し、生成されたメトリクスをログします。

<Tabs
  defaultValue="torch"
  values={[
    {label: 'TorchModel', value: 'torch'},
    {label: 'KerasModel', value: 'keras'},
  ]}>
  <TabItem value="torch">

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # オプション
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```
  </TabItem>
  <TabItem value="keras">

```python
from deepchem.models import KerasModel, ValidationCallback
vc = ValidationCallback(…)  # 任意

model = KerasModel(…, wandb_logger=logger)

model.fit(…, callbacks=[vc])

logger.finish()

```
</TabItem>

</Tabs>