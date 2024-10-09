---
title: Track Jupyter notebooks
description: Jupyter에서 W&B를 사용하여 노트북을 떠나지 않고도 대화형 시각화를 얻으세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B를 Jupyter와 함께 사용하여 노트북을 벗어나지 않고도 인터랙티브한 시각화를 얻을 수 있습니다. 맞춤형 분석, Experiments 및 프로토타입을 결합하여 모두 완전하게 기록됩니다!

## Jupyter 노트북과 함께 W&B를 사용하는 유스 케이스

1. **반복적인 실험**: 실험을 실행하고 재실행하면서 파라미터를 조정하고, 여러분이 수행하는 모든 Runs를 자동으로 W&B에 저장하여, 중간에 수동으로 노트를 작성할 필요가 없습니다.
2. **코드 저장**: 모델을 재현할 때, 노트북에서 어떤 셀이 실행되었고 어떤 순서로 실행되었는지 알기 어려울 수 있습니다. [설정 페이지](../app/settings-page/intro.md)에서 코드 저장을 활성화하여, 각 실험에 대한 셀 실행 기록을 저장하세요.
3. **맞춤형 분석**: Runs가 W&B에 기록되면 API를 통해 쉽게 데이터프레임을 가져와서 맞춤형 분석을 수행하고, 그런 다음 그 결과를 W&B에 기록하여 Reports에서 저장하고 공유할 수 있습니다.

## 노트북에서 시작하기

다음 코드를 사용하여 노트북을 시작하고 W&B를 설치하며 계정을 연결하세요:

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

`wandb.init()`을 실행한 후에는 `%%wandb`로 새로운 셀을 시작하여 노트북에서 실시간 그래프를 확인하세요. 이 셀을 여러 번 실행하면 데이터가 run에 추가됩니다.

```notebook
%%wandb

# 여기에 트레이닝 루프를 넣으세요
```

[빠른 예제 노트북 →](http://wandb.me/jupyter-interact-colab)에서 직접 시도해 보세요.

![](/images/track/jupyter_widget.png)

### 노트북 내에서 직접 라이브 W&B 인터페이스 렌더링

`%wandb` 매직을 사용하여 기존 대시보드, Sweeps 또는 Reports를 노트북 내에 직접 표시할 수도 있습니다:

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

`%%wandb` 또는 `%wandb` 매직의 대안으로, `wandb.init()`을 실행한 후 셀을 `wandb.run`으로 끝내어 인라인 그래프를 표시하거나, api에서 반환된 모든 리포트, 스윕 또는 run 오브젝트에 대해 `ipython.display(...)`를 호출할 수 있습니다.

```python
# 먼저 wandb.run을 초기화하세요
wandb.init()

# 셀 출력이 wandb.run인 경우, 실시간 그래프를 볼 수 있습니다
wandb.run
```

:::info
W&B로 할 수 있는 것들에 대해 더 알고 싶나요? [데이터와 미디어 로깅 가이드](log/intro.md)를 확인해 보거나, [자신의 좋아하는 ML 툴킷과의 인테그레이션 방법](../integrations/intro.md)을 배우거나, 단순히 [레퍼런스 문서](../../ref/python/README.md)나 [예제 레포](https://github.com/wandb/examples)를 바로 탐색해 보세요.
:::

## W&B의 추가적인 Jupyter 기능

1. **Colab에서의 간편한 인증**: Colab에서 처음 `wandb.init`를 호출할 때, 현재 브라우저에서 W&B에 로그인되어 있다면, 런타임을 자동으로 인증합니다. run 페이지의 Overview 탭에서 Colab으로 연결되는 링크를 볼 수 있습니다.
2. **Jupyter Magic:** 대시보드, Sweeps 및 Reports를 노트북에 직접 표시하세요. `%wandb` 매직은 프로젝트, Sweeps 또는 Reports 경로를 받아 W&B 인터페이스를 노트북에 직접 렌더링합니다.
3. **도커화된 Jupyter 실행**: `wandb docker --jupyter`를 호출하여 docker 컨테이너를 실행하고, 거기에 코드를 마운트하며 Jupyter가 설치되도록 하고 포트 8888에서 실행하세요.
4. **임의의 순서로 셀 실행**: 기본적으로, 다음 `wandb.init`가 호출될 때까지 Run을 "완료" 상태로 표시하지 않습니다. 이를 통해 여러 개의 셀(예: 데이터 설정 셀 하나, 트레이닝 셀 하나, 테스트 셀 하나)을 원하는 순서로 실행하고 모두 동일한 Run에 로그할 수 있습니다. [설정](https://app.wandb.ai/settings)에서 코드 저장을 켜면, 실행된 셀들을 실행된 순서와 상태로 로그하며, 가장 비선형적인 파이프라인조차 재현할 수 있게 됩니다. Jupyter 노트북에서 수동으로 Run을 완료 상태로 표시하려면, `run.finish`를 호출하세요.

```python
import wandb

run = wandb.init()

# 트레이닝 스크립트와 로그가 여기에 들어갑니다

run.finish()
```

## 자주 묻는 질문

### W&B 안내 메시지를 어떻게 숨길 수 있나요?

표준 wandb 로깅 및 안내 메시지(예: Run 시작 시 프로젝트 정보)를 비활성화하려면 `wandb.login` 실행 _전_에 노트북 셀에 다음을 실행하세요:

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

노트북에 `INFO SenderThread:11484 [sender.py:finish():979]` 같은 로그 메시지가 표시되는 경우, 다음을 사용하여 비활성화할 수 있습니다:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

### `WANDB_NOTEBOOK_NAME`을 어떻게 설정하나요?

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` 오류 메시지를 보는 경우, 환경 변수를 설정하여 문제가 해결할 수 있습니다. 여러 방법이 있습니다:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env "WANDB_NOTEBOOK_NAME" "여기에 노트북 이름"
```
  </TabItem>
  <TabItem value="python">

```notebook
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "여기에 노트북 이름"
```
  </TabItem>
</Tabs>