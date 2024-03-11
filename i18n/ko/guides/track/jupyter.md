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

W&B를 Jupyter와 함께 사용하여 노트북을 벗어나지 않고도 인터랙티브한 시각화를 얻으십시오. 맞춤형 분석, 실험 그리고 프로토타입을 결합하여 모두 완벽하게 로그하세요!

## Jupyter 노트북으로 W&B 사용 사례

1. **반복적인 실험**: 실험을 실행하고 다시 실행하며, 파라미터를 조정하고 수동으로 메모를 하지 않고도 W&B에 자동으로 저장되는 모든 run을 가집니다.
2. **코드 저장**: 모델을 재현할 때, 노트북의 어떤 셀이 어떤 순서로 실행되었는지 알기 어렵습니다. [설정 페이지](../app/settings-page/intro.md)에서 코드 저장을 켜서 각 실험에 대한 셀 실행 기록을 저장하세요.
3. **맞춤형 분석**: run이 W&B에 로그되면, API에서 데이터프레임을 쉽게 얻고 맞춤형 분석을 수행한 다음, 그 결과를 W&B에 로그하여 리포트에 저장하고 공유할 수 있습니다.

## 노트북에서 시작하기

다음 코드로 노트북을 시작하여 W&B를 설치하고 계정을 연결하세요:

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

`wandb.init()`를 실행한 후, `%%wandb`로 시작하는 새 셀로 노트북에서 실시간 그래프를 볼 수 있습니다. 이 셀을 여러 번 실행하면, 데이터가 run에 추가됩니다.

```notebook
%%wandb

# 여기에 트레이닝 루프를 입력하세요
```

이 [빠른 예제 노트북 →](http://wandb.me/jupyter-interact-colab)에서 직접 시도해 보세요.

![](/images/track/jupyter_widget.png)

### 노트북에서 직접 라이브 W&B 인터페이스 렌더링하기

`%wandb` 매직을 사용하여 기존 대시보드, 스윕 또는 리포트를 노트북에 직접 표시할 수도 있습니다:

```notebook
# 프로젝트 워크스페이스 표시
%wandb USERNAME/PROJECT
# 단일 run 표시
%wandb USERNAME/PROJECT/runs/RUN_ID
# 스윕 표시
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# 리포트 표시
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 임베디드 iframe의 높이 지정
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` 또는 `%wandb` 매직 대신에, `wandb.init()`을 실행한 후에는 `wandb.run`을 셀의 끝에 넣어 인라인 그래프를 표시하거나, 우리 API에서 반환된 리포트, 스윕 또는 run 오브젝트에 대해 `ipython.display(...)`를 호출할 수 있습니다.

```python
# wandb.run을 먼저 초기화하세요
wandb.init()

# 셀이 wandb.run을 출력하면, 실시간 그래프를 볼 수 있습니다
wandb.run
```

:::info
W&B를 사용하여 무엇을 할 수 있는지 더 알고 싶으신가요? [데이터 및 미디어 로깅 가이드](log/intro.md)를 확인하거나, [당신의 좋아하는 ML 툴킷과 우리를 통합하는 방법](../integrations/intro.md)을 배우거나, 바로 [참조 문서](../../ref/python/README.md)나 [예제 레포](https://github.com/wandb/examples)에 뛰어들어 보세요.
:::

## W&B에서의 추가 Jupyter 기능들

1. **Colab에서 쉬운 인증**: Colab에서 처음으로 `wandb.init`을 호출하면, 현재 브라우저에서 W&B에 로그인되어 있다면 자동으로 런타임을 인증합니다. run 페이지의 Overview 탭에서 Colab에 대한 링크를 볼 수 있습니다.
2. **Jupyter Magic:** 대시보드, 스윕 및 리포트를 직접 노트북에 표시합니다. `%wandb` 매직은 프로젝트, 스윕 또는 리포트로의 경로를 받아 들이고 W&B 인터페이스를 노트북에 직접 렌더링합니다.
3. **도커화된 Jupyter 실행**: `wandb docker --jupyter`를 호출하여 docker 컨테이너를 시작하고, 코드를 마운트하고, Jupyter가 설치되어 있고 8888 포트에서 시작하도록 합니다.
4. **두려움 없이 임의의 순서로 셀 실행**: 기본적으로, 우리는 다음 `wandb.init`이 호출될 때까지 run을 "완료됨"으로 표시하는 것을 기다립니다. 이를 통해 여러 셀(예: 데이터 설정, 트레이닝, 테스트를 위한 것)을 원하는 순서대로 실행하고 모두 동일한 run에 로그할 수 있습니다. [설정](https://app.wandb.ai/settings)에서 코드 저장을 켜면, 실행된 셀도 순서대로, 실행된 상태로 로그하여 가장 비선형적인 파이프라인조차도 재현할 수 있습니다. Jupyter 노트북에서 수동으로 run을 완료하려면 `run.finish`를 호출하세요.

```python
import wandb

run = wandb.init()

# 트레이닝 스크립트와 로깅은 여기에 입력하세요

run.finish()
```

## 자주 묻는 질문

### W&B 정보 메시지를 어떻게 무음 처리하나요?

표준 wandb 로깅 및 정보 메시지(예: run 시작 시의 프로젝트 정보)를 비활성화하려면, `wandb.login`을 실행하기 _전에_ 노트북 셀에서 다음을 실행하세요:

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

노트북에서 `안내 SenderThread:11484 [sender.py:finish():979]`와 같은 로그 메시지가 보이면, 다음을 사용하여 비활성화할 수 있습니다:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

### `WANDB_NOTEBOOK_NAME`을 어떻게 설정하나요?

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` 오류 메시지가 보이면, 환경 변수를 설정하여 해결할 수 있습니다. 설정하는 방법은 여러 가지가 있습니다:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
  </TabItem>
  <TabItem value="python">

```notebook
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
  </TabItem>
</Tabs>