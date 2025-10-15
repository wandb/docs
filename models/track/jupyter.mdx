---
title: Track Jupyter notebooks
description: Jupyter 와 W&B를 함께 사용하여 노트북을 벗어나지 않고도 대화형 시각화 를 얻으세요.
menu:
  default:
    identifier: ko-guides-models-track-jupyter
    parent: experiments
weight: 6
---

Jupyter와 W&B를 함께 사용하면 노트북에서 나가지 않고도 대화형 시각화 자료를 얻을 수 있습니다. 사용자 정의 분석, Experiments, 프로토타입을 모두 결합하여 완벽하게 로그할 수 있습니다.

## Jupyter 노트북과 W&B의 유스 케이스

1. **반복적인 실험**: 파라미터를 조정하여 Experiments를 실행하고 다시 실행하면 수동으로 메모하지 않아도 수행하는 모든 Runs가 자동으로 W&B에 저장됩니다.
2. **코드 저장**: 모델을 재현할 때 노트북에서 어떤 셀이 어떤 순서로 실행되었는지 알기 어렵습니다. [설정 페이지]({{< relref path="/guides/models/app/settings-page/" lang="ko" >}})에서 코드 저장을 켜면 각 experiment에 대한 셀 실행 기록이 저장됩니다.
3. **사용자 정의 분석**: Runs가 W&B에 기록되면 API에서 데이터프레임을 가져와 사용자 정의 분석을 수행한 다음 해당 결과를 W&B에 기록하여 리포트에 저장하고 공유하기 쉽습니다.

## 노트북에서 시작하기

다음 코드로 노트북을 시작하여 W&B를 설치하고 계정을 연결합니다.

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

다음으로 experiment를 설정하고 하이퍼파라미터를 저장합니다.

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

`wandb.init()`를 실행한 후 `%%wandb`로 새 셀을 시작하여 노트북에서 라이브 그래프를 확인합니다. 이 셀을 여러 번 실행하면 데이터가 Run에 추가됩니다.

```notebook
%%wandb

# Your training loop here
```

[예제 노트북](http://wandb.me/jupyter-interact-colab)에서 직접 사용해 보세요.

{{< img src="/images/track/jupyter_widget.png" alt="" >}}

### 노트북에서 직접 라이브 W&B 인터페이스 렌더링

`%wandb` 매직을 사용하여 기존의 대시보드, Sweeps 또는 리포트를 노트북에서 직접 표시할 수도 있습니다.

```notebook
# Display a project workspace
%wandb USERNAME/PROJECT
# Display a single run
%wandb USERNAME/PROJECT/runs/RUN_ID
# Display a sweep
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# Display a report
%wandb USERNAME/PROJECT/reports/REPORT_ID
# Specify the height of embedded iframe
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` 또는 `%wandb` 매직의 대안으로 `wandb.init()`를 실행한 후 `wandb.run`으로 셀을 종료하여 인라인 그래프를 표시하거나 API에서 반환된 리포트, 스윕 또는 Run 오브젝트에서 `ipython.display(...)`를 호출할 수 있습니다.

```python
# Initialize wandb.run first
wandb.init()

# If cell outputs wandb.run, you'll see live graphs
wandb.run
```

{{% alert %}}
W&B로 무엇을 할 수 있는지 더 자세히 알고 싶으신가요? [데이터 및 미디어 로깅 가이드]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 확인하거나, [자주 사용하는 ML 툴킷과 통합하는 방법]({{< relref path="/guides/integrations/" lang="ko" >}})을 배우거나, [참조 문서]({{< relref path="/ref/python/" lang="ko" >}}) 또는 [예제 리포지토리](https://github.com/wandb/examples)로 바로 이동하세요.
{{% /alert %}}

## W&B의 추가 Jupyter 기능

1. **Colab에서 간편한 인증**: Colab에서 `wandb.init`를 처음 호출하면 브라우저에서 W&B에 로그인한 경우 런타임이 자동으로 인증됩니다. Run 페이지의 Overview 탭에서 Colab 링크를 볼 수 있습니다.
2. **Jupyter Magic**: 대시보드, Sweeps 및 리포트를 노트북에서 직접 표시합니다. `%wandb` 매직은 프로젝트, Sweeps 또는 리포트의 경로를 허용하고 W&B 인터페이스를 노트북에서 직접 렌더링합니다.
3. **dockerized Jupyter 실행**: `wandb docker --jupyter`를 호출하여 docker 컨테이너를 시작하고, 코드를 마운트하고, Jupyter가 설치되었는지 확인하고, 포트 8888에서 시작합니다.
4. **두려움 없이 임의의 순서로 셀 실행**: 기본적으로 Run이 `finished`로 표시될 때까지 `wandb.init`가 다음에 호출될 때까지 기다립니다. 이를 통해 여러 셀(예: 데이터 설정, 트레이닝, 테스트)을 원하는 순서로 실행하고 모두 동일한 Run에 기록할 수 있습니다. [설정](https://app.wandb.ai/settings)에서 코드 저장을 켜면 실행된 셀도 순서대로, 그리고 실행된 상태로 기록하여 가장 비선형적인 파이프라인도 재현할 수 있습니다. Jupyter 노트북에서 Run을 수동으로 완료하려면 `run.finish`를 호출합니다.

```python
import wandb

run = wandb.init()

# training script and logging goes here

run.finish()
```