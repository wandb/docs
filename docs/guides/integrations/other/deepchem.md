---
title: DeepChem
description: W&B를 DeepChem 라이브러리와 통합하는 방법.
slug: /guides/integrations/deepchem
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[DeepChem 라이브러리](https://github.com/deepchem/deepchem)는 약물 발견, 재료 과학, 화학 및 생물학에서 딥러닝 사용을 민주화하는 오픈 소스 툴을 제공합니다. 이 W&B 인테그레이션은 DeepChem을 사용하여 모델을 트레이닝할 때 실험 추적 및 모델 체크포인트를 간단하고 사용하기 쉽게 추가합니다.

## 🧪 3줄 코드로 DeepChem 로그 설정

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

![](/images/integrations/cd.png)

## 리포트 및 Google Colab

W&B DeepChem 인테그레이션을 사용하여 생성된 예제 차트를 보려면 [W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) 문서를 탐색하세요.

바로 작동하는 코드를 살펴보고 싶다면, 이 [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb)을 확인하세요.

## 시작하기: 실험 추적

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) 또는 [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 유형의 DeepChem 모델용으로 Weights & Biases를 설정하세요.

### 1) `wandb` 라이브러리 설치 및 로그인

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

### 2) WandbLogger 초기화 및 구성

```python
from deepchem.models import WandbLogger

logger = WandbLogger(entity="my_entity", project="my_project")
```

### 3) 트레이닝 및 평가 데이터를 W&B에 로그하기

트레이닝 손실 및 평가 메트릭을 Weights & Biases에 자동으로 로그할 수 있습니다. 선택적 평가 기능은 DeepChem [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py)을 사용하여 활성화할 수 있으며, `WandbLogger`는 ValidationCallback 콜백을 감지하고 생성된 메트릭을 로그합니다.

<Tabs
  defaultValue="torch"
  values={[
    {label: 'TorchModel', value: 'torch'},
    {label: 'KerasModel', value: 'keras'},
  ]}>
  <TabItem value="torch">

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # 선택 사항
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```
  </TabItem>
  <TabItem value="keras">

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # 선택 사항
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

  </TabItem>
</Tabs>