---
title: DeepChem
description: DeepChem 라이브러리 와 W&B 를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-deepchem
    parent: integrations
weight: 70
---

[DeepChem 라이브러리](https://github.com/deepchem/deepchem)는 약물 발견, 재료 과학, 화학 및 생물학에서 딥러닝 사용을 대중화하는 오픈 소스 툴을 제공합니다. 이 W&B 인테그레이션은 DeepChem을 사용하여 모델을 트레이닝하는 동안 간단하고 사용하기 쉬운 experiment 추적 및 모델 체크포인팅을 추가합니다.

## 3줄의 코드로 DeepChem 로깅하기

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="" >}}

## Report 및 Google Colab

[W&B와 DeepChem 사용: 분자 그래프 컨볼루션 네트워크](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) 아티클에서 W&B DeepChem 인테그레이션을 사용하여 생성된 차트 예제를 살펴보세요.

작동하는 코드로 바로 들어가려면 이 [**Google Colab**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb)을 확인하세요.

## Experiments 추적

[KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) 또는 [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 유형의 DeepChem 모델에 대해 W&B를 설정합니다.

### 가입하고 API 키 만들기

API 키는 W&B에 대한 컴퓨터를 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장합니다.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

### `wandb` 라이브러리를 설치하고 로그인하기

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면 다음을 수행합니다.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{< /tabpane >}}

### W&B에 트레이닝 및 평가 데이터 기록하기

트레이닝 손실 및 평가 메트릭은 W&B에 자동으로 기록될 수 있습니다. DeepChem [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py)을 사용하여 선택적 평가를 활성화할 수 있습니다. `WandbLogger`는 ValidationCallback 콜백을 감지하고 생성된 메트릭을 기록합니다.

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # optional
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}
