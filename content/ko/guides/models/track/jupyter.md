---
title: Jupyter 노트북 추적하기
description: Jupyter에서 W&B를 사용하면 노트북을 벗어나지 않고도 대화형 시각화를 바로 얻을 수 있습니다.
menu:
  default:
    identifier: ko-guides-models-track-jupyter
    parent: experiments
weight: 6
---

Jupyter에서 W&B를 사용하면 노트북을 벗어나지 않고 대화형 시각화 기능을 사용할 수 있습니다. 커스텀 분석, 실험, 프로토타입 작업을 모두 완벽하게 로그로 남기면서 결합할 수 있습니다.

## Jupyter 노트북에서의 W&B 주요 활용법

1. **반복 실험**: 실험을 반복 실행하고 파라미터를 조정할 때마다 각 실행(run)이 따로 기록될 필요 없이, 모든 run이 W&B에 자동 저장됩니다.
2. **코드 저장**: 모델 재현이 필요할 때 노트북의 셀 실행 위치와 순서를 파악하기 어렵습니다. [설정 페이지]({{< relref path="/guides/models/app/settings-page/" lang="ko" >}})에서 코드 저장을 켜면, 실험마다 셀 실행 이력을 저장할 수 있습니다.
3. **커스텀 분석**: 여러 run을 W&B에 로깅하면 API를 통해 손쉽게 데이터프레임으로 가져와 커스텀 분석이 가능합니다. 분석 결과 역시 W&B에 다시 로깅해 리포트로 저장·공유할 수 있습니다.

## 노트북에서 시작하기

아래 코드를 실행해 노트북에서 W&B를 설치하고 계정을 연동하세요:

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

이후 실험을 초기화하고 하이퍼파라미터를 저장하세요:

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

`wandb.init()` 실행 후 새로운 셀에서 `%%wandb`를 입력하면 노트북 내에서 실시간 그래프를 볼 수 있습니다. 이 셀을 여러 번 실행해도 데이터가 동일한 run에 계속 추가됩니다.

```notebook
%%wandb

# 여기에 트레이닝 루프를 작성하세요
```

[예제 노트북](https://wandb.me/jupyter-interact-colab)에서 직접 체험해보세요.

{{< img src="/images/track/jupyter_widget.png" alt="Jupyter W&B 위젯" >}}

### 노트북에서 W&B 인터페이스 바로 렌더링하기

`%wandb` 매직 명령어를 사용하면 기존의 dashboard, sweep, report를 노트북 내에서 바로 렌더링할 수 있습니다:

```notebook
# 프로젝트 워크스페이스 표시
%wandb USERNAME/PROJECT
# 특정 run 표시
%wandb USERNAME/PROJECT/runs/RUN_ID
# sweep 표시
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# report 표시
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 임베드되는 iframe 높이 설정
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` 또는 `%wandb` 매직 명령어 대신, `wandb.init()` 실행 후 셀 마지막에 `wandb.Run.finish()`를 호출하면 인라인 그래프를 볼 수 있습니다. 또는 API로부터 받은 report, sweep, run 오브젝트에 대해 `ipython.display(...)`로 직접 렌더링할 수도 있습니다.

```python
import wandb
from IPython.display import display
# run 초기화
run = wandb.init()

# 셀에서 run.finish()를 실행하면 실시간 그래프가 표시됩니다
run.finish()
```

{{% alert %}}
W&B의 다양한 활용 방법이 궁금하다면? [데이터 및 미디어 로깅 가이드]({{< relref path="/guides/models/track/log/" lang="ko" >}})를 참고하거나, [다양한 ML 툴킷과의 인테그레이션 방법]({{< relref path="/guides/integrations/" lang="ko" >}})을 확인해보세요. 아니면 바로 [레퍼런스 문서]({{< relref path="/ref/python/" lang="ko" >}})와 [예제 저장소](https://github.com/wandb/examples)도 둘러볼 수 있습니다.
{{% /alert %}}

## W&B에서 지원하는 추가 Jupyter 기능

1. **Colab에서 간편 인증:** Colab 환경에서 `wandb.init`을 처음 실행할 때, 브라우저에 W&B 로그인이 되어 있다면 런타임이 자동 인증됩니다. run 페이지 Overview 탭에서 Colab 링크도 확인 가능합니다.
2. **Jupyter 매직:** dashboard, sweep, report를 노트북 내에서 바로 볼 수 있습니다. `%wandb` 매직 명령에 프로젝트, sweep, report 경로를 넣으면 W&B 인터페이스가 노트북 안에서 바로 렌더링됩니다.
3. **도커로 Jupyter 실행:** `wandb docker --jupyter` 명령을 통해 Docker 컨테이너에서 Jupyter를 즉시 실행하고 코드 마운트와 Jupyter 설치, 8888 포트 접근이 자동으로 처리됩니다.
4. **셀 실행 순서 자유:** 다음번 `wandb.init` 전까지 run이 `finished` 상태로 바뀌지 않습니다. 데이터 준비, 트레이닝, 테스트 등 셀을 원하는 순서로 실행해도 동일 run에 계속 기록됩니다. [설정](https://app.wandb.ai/settings)에서 코드 저장을 활성화하면, 셀 실행 정보를 실행 순서와 함께 저장해 비선형 파이프라인도 안전하게 재현할 수 있습니다. Jupyter 노트북에서 run을 수동으로 끝내려면 `run.finish`를 호출하세요.

```python
import wandb

run = wandb.init()

# 트레이닝 스크립트와 로그 코드 작성

run.finish()
```