---
description: se W&B with Jupyter to get interactive visualizations without leaving
  your notebook.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Jupyter 노트북 추적하기

<head>
  <title>Jupyter 노트북 추적하기</title>
</head>

W&B와 Jupyter를 사용하여 노트북을 떠나지 않고도 인터랙티브한 시각화를 얻을 수 있습니다. 맞춤 분석, 실험 및 프로토타입을 결합하여 모두 완벽하게 기록하십시오!

## Jupyter 노트북에서 W&B 사용 사례

1. **반복적인 실험**: 실험을 실행하고 다시 실행하며 파라미터를 조정하고, 수동으로 메모를 하지 않아도 W&B에 자동으로 모든 실행이 저장됩니다.
2. **코드 저장**: 모델을 재현할 때 노트북의 어떤 셀이 어떤 순서로 실행되었는지 알기 어렵습니다. [설정 페이지](../app/settings-page/intro.md)에서 코드 저장을 활성화하여 각 실험에 대한 셀 실행 기록을 저장하세요.
3. **맞춤 분석**: 실행이 W&B에 로그되면 API에서 데이터프레임을 쉽게 얻고 맞춤 분석을 수행한 다음 그 결과를 W&B에 로그하여 리포트에서 저장하고 공유할 수 있습니다.

## 노트북에서 시작하기

다음 코드로 노트북을 시작하여 W&B를 설치하고 계정을 연결하십시오:

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

다음으로, 실험을 설정하고 하이퍼파라미터를 저장하세요:

```python
wandb.init(
    project="jupyter-projo",
    config={
        "batch_size": 128,
        "learning_rate": 0.01,
        "dataset": "CIFAR-100",
    },
)
```

`wandb.init()`를 실행한 후 새 셀에서 `%%wandb`로 시작하여 노트북에서 실시간 그래프를 볼 수 있습니다. 이 셀을 여러 번 실행하면 실행에 데이터가 추가됩니다.

```notebook
%%wandb

# 여기에 학습 루프
```

이 [빠른 예제 노트북 →](http://wandb.me/jupyter-interact-colab)에서 직접 시도해 보세요.

![](/images/track/jupyter_widget.png)

### 노트북에서 직접 실시간 W&B 인터페이스 렌더링하기

`%%wandb` 또는 `%wandb` 매직을 대신하여 `wandb.init()`을 실행한 후에는 어떤 셀에서든 `wandb.run`으로 끝내면 인라인 그래프를 볼 수 있거나, 우리의 API에서 반환된 어떤 리포트, 스윕 또는 실행 개체에 대해 `ipython.display(...)`를 호출할 수 있습니다.

```python
# 먼저 wandb.run을 초기화합니다
wandb.init()

# 셀 출력이 wandb.run이면 실시간 그래프를 볼 수 있습니다
wandb.run
```

:::info
W&B로 할 수 있는 것에 대해 더 알고 싶으십니까? [데이터 및 미디어 로깅 가이드](log/intro.md)를 확인하고, [가장 좋아하는 ML 툴킷과 통합하는 방법](../integrations/intro.md)을 배우거나, 바로 [참조 문서](../../ref/python/README.md)나 [예제 리포지토리](https://github.com/wandb/examples)로 뛰어들어보세요.
:::

## W&B의 추가 Jupyter 기능

1. **Colab에서 쉬운 인증**: Colab에서 처음으로 `wandb.init`을 호출할 때 현재 브라우저에서 W&B에 로그인되어 있다면 자동으로 런타임을 인증합니다. 실행 페이지의 Overview 탭에서 Colab에 대한 링크를 볼 수 있습니다.
2. **Jupyter Magic:** 대시보드, 스윕 및 리포트를 노트북에 직접 표시합니다. `%wandb` 매직은 프로젝트, 스윕 또는 리포트로의 경로를 받아들이고 노트북 내에서 직접 W&B 인터페이스를 렌더링합니다.
3. **도커화된 Jupyter 실행**: 코드를 마운트하고, Jupyter가 설치되어 있으며, 포트 8888에서 실행되는 도커 컨테이너를 시작하려면 `wandb docker --jupyter`를 호출하세요.
4. **임의의 순서로 셀 실행하기**: 기본적으로, 우리는 `wandb.init`이 다음에 호출될 때까지 실행을 "완료"로 표시하는 것을 기다립니다. 이를 통해 여러 셀(예: 데이터 설정, 학습, 테스트를 위한 셀)을 원하는 순서대로 실행하고 모두 동일한 실행에 로그할 수 있습니다. [설정](https://app.wandb.ai/settings)에서 코드 저장을 켜면 실행된 셀도 순서대로, 실행된 상태로 로그하여 가장 비선형적인 파이프라인도 재현할 수 있습니다. Jupyter 노트북에서 실행을 수동으로 완료하려면 `run.finish`를 호출하세요.

```python
import wandb

run = wandb.init()

# 여기에 학습 스크립트와 로깅이 있습니다

run.finish()
```

## 자주 묻는 질문

### W&B 정보 메시지를 어떻게 끌 수 있나요?

표준 wandb 로깅 및 정보 메시지(예: 실행 시작 시 프로젝트 정보)를 비활성화하려면 `wandb.login`을 실행하기 _전에_ 노트북 셀에서 다음을 실행하세요:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'Python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env WANDB_SILENT=True
```
  </TabItem>
  <TabItem value="python">

```python
import os

os.environ["WANDB_SILENT"] = "True"
```
  </TabItem>
</Tabs>

노트북에서 `INFO SenderThread:11484 [sender.py:finish():979]`와 같은 로그 메시지를 본다면, 다음을 사용하여 해당 메시지를 비활성화할 수 있습니다:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

### `WANDB_NOTEBOOK_NAME`을 어떻게 설정하나요?

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` 오류 메시지가 표시되면 환경 변수를 설정하여 해결할 수 있습니다. 여러 방법이 있습니다:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env "WANDB_NOTEBOOK_NAME" "노트북 이름 여기에"
```
  </TabItem>
  <TabItem value="python">

```notebook
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "노트북 이름 여기에"
```
  </TabItem>
</Tabs>