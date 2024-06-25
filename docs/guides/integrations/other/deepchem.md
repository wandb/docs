---
description: DeepChemライブラリとW&Bを統合する方法
slug: /guides/integrations/deepchem
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# DeepChem

[DeepChem library](https://github.com/deepchem/deepchem) は、創薬、材料科学、化学、生物学におけるディープラーニングの利用を民主化するオープンソースツールを提供します。この W&B インテグレーションは、DeepChem を使用してモデルをトレーニングする際に、シンプルで使いやすい実験管理とモデルチェックポイントを追加します。

## 🧪 3行のコードで DeepChem ログを記録

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

![](@site/static/images/integrations/cd.png)

## レポート & Google Colab

W&B DeepChem インテグレーションを使用して生成されたチャートの例については、[Using W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=)の記事をご覧ください。

実際のコードにすぐに取り組みたい場合は、この[**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb)をチェックしてください。

## はじめに: Experiments をトラッキングする

[Weights & Biases](https://wandb.ai/site) を DeepChem モデルで使用するには、[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) または [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 型のモデルを設定します。

### 1) `wandb` ライブラリをインストールしてログイン

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
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

### 2) WandbLogger を初期化して設定

```python
from deepchem.models import WandbLogger

logger = WandbLogger(entity="my_entity", project="my_project")
```

### 3) トレーニングと評価データを W&B にログ

トレーニングロスや評価メトリクスは自動的に Weights & Biases にログできます。オプションの評価は、DeepChem の [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py) を使用して有効化できます。`WandbLogger` は ValidationCallback コールバックを検出し、生成されたメトリクスをログします。

<Tabs
  defaultValue="torch"
  values={[
    {label: 'TorchModel', value: 'torch'},
    {label: 'KerasModel', value: 'keras'},
  ]}>
  <TabItem value="torch">

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```
  </TabItem>
  <TabItem value="keras">

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

  </TabItem>
</Tabs>
