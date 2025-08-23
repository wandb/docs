---
title: DeepChem
description: W&B를 DeepChem 라이브러리와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-deepchem
    parent: integrations
weight: 70
---

[DeepChem 라이브러리](https://github.com/deepchem/deepchem)는 오픈 소스 툴을 제공하여 신약 개발, 재료 과학, 화학, 생물학 분야에서 딥러닝 활용을 누구나 쉽게 할 수 있도록 지원합니다. 이 W&B 인테그레이션은 DeepChem으로 모델을 트레이닝할 때 간단하고 손쉬운 실험 트래킹과 모델 체크포인트 기능을 추가해줍니다.

## 3줄 코드로 DeepChem 로그 남기기

```python
logger = WandbLogger(…)
model = TorchModel(…, wandb_logger=logger)
model.fit(…)
```

{{< img src="/images/integrations/cd.png" alt="DeepChem 분자 분석" >}}

## 리포트 및 Google Colab

W&B DeepChem 인테그레이션으로 생성한 예제 차트를 보려면 [Using W&B with DeepChem: Molecular Graph Convolutional Networks](https://wandb.ai/kshen/deepchem_graphconv/reports/Using-W-B-with-DeepChem-Molecular-Graph-Convolutional-Networks--Vmlldzo4MzU5MDc?galleryTag=) 글을 참고해보세요.

실제로 작동하는 코드를 바로 확인하고 싶다면, 이 [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/deepchem/W%26B_x_DeepChem.ipynb) 데모를 이용해보세요.

## 실험 트래킹하기

W&B를 [KerasModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#keras-models) 또는 [TorchModel](https://deepchem.readthedocs.io/en/latest/api_reference/models.html#pytorch-models) 타입 DeepChem 모델에 설정할 수 있습니다.

### 회원가입 및 API 키 생성

API 키는 W&B에 내 컴퓨터를 인증해주는 역할을 합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 간편하게 API 키를 발급 받으려면 [W&B 인증 페이지](https://wandb.ai/authorize)에 바로 접속하세요. 화면에 표시되는 API 키를 복사하여 패스워드 매니저 등 안전한 곳에 보관하세요.
{{% /alert %}}

1. 오른쪽 상단에서 사용자 프로필 아이콘을 클릭합니다.
1. **User Settings**를 선택하고, 아래로 스크롤해 **API Keys** 섹션으로 이동합니다.
1. **Reveal** 버튼을 클릭해 표시된 API 키를 복사하세요. 키를 다시 숨기려면 페이지를 새로고침하세요.

### `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하는 방법:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 본인의 API 키로 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.



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

{{% tab header="Python 노트북" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}

{{< /tabpane >}}

### 트레이닝 및 평가 데이터를 W&B에 로그하기

트레이닝 손실값과 평가 메트릭은 W&B에 자동으로 로그될 수 있습니다. 선택적으로 DeepChem의 [ValidationCallback](https://github.com/deepchem/deepchem/blob/master/deepchem/models/callbacks.py)을 사용하면, `WandbLogger`가 ValidationCallback 콜백을 감지하여 생성된 메트릭도 함께 로그합니다.

{{< tabpane text=true >}}

{{% tab header="TorchModel" value="torch" %}}

```python
from deepchem.models import TorchModel, ValidationCallback

vc = ValidationCallback(…)  # 선택 사항
model = TorchModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{% tab header="KerasModel" value="keras" %}}

```python
from deepchem.models import KerasModel, ValidationCallback

vc = ValidationCallback(…)  # 선택 사항
model = KerasModel(…, wandb_logger=logger)
model.fit(…, callbacks=[vc])
logger.finish()
```

{{% /tab %}}

{{< /tabpane >}}