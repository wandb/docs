---
description: How to integrate W&B with DeepChem library.
slug: /guides/integrations/deepchem
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# DeepChem

[DeepChem 라이브러리](https://github.com/deepchem/deepchem)는 약물 발견, 재료 과학, 화학, 생물학에서 딥러닝의 사용을 민주화하는 오픈 소스 툴을 제공합니다. 이 Weights & Biases 인테그레이션은 DeepChem을 사용하여 모델을 트레이닝할 때 간단하고 쉽게 사용할 수 있는 실험 추적과 모델 체크포인트를 추가합니다.

## 🧪 DeepChem에서 3줄의 코드로 로깅하기

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

![](@site/static/images/integrations/cd.png)

## 리포트 & Google Colab

W&B DeepChem 인테그레이션을 사용하여 생성된 예제 차트를 살펴보려면 [W&B와 DeepChem 사용하기: 분자 그래프 컨볼루셔널 네트워크](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) 기사를 탐색하세요.

바로 작동하는 코드로 뛰어들고 싶다면 이 [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb)을 확인하세요.

## 시작하기: 실험 추적하기

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) 또는 [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 타입의 DeepChem 모델을 위해 Weights & Biases를 설정하세요.

### 1) `wandb` 라이브러리를 설치하고 로그인하기

<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'},
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

### 2) WandbLogger 초기화 및 설정하기

```python
from deepchem.models import WandbLogger

logger = WandbLogger(entity="my_entity", project="my_project")
```

### 3) 트레이닝 및 평가 데이터를 W&B에 로그하기

트레이닝 손실과 평가 메트릭은 Weights & Biases에 자동으로 로그될 수 있습니다. 선택적 평가는 DeepChem의 [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py)을 사용하여 활성화할 수 있으며, `WandbLogger`는 ValidationCallback 콜백을 감지하고 생성된 메트릭을 로그할 것입니다.

<Tabs
  defaultValue="torch"
  values={[
    {label: 'TorchModel', value: 'torch'},
    {label: 'KerasModel', value: 'keras'},
  ]}>
  <TabItem value="torch">

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # 선택적
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```
  </TabItem>
  <TabItem value="keras">

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # 선택적
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

  </TabItem>
</Tabs>