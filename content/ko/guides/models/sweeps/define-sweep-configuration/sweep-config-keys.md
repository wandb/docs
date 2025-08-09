---
title: 스윕 구성 옵션
menu:
  default:
    identifier: ko-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

스윕 구성은 중첩된 키-값 쌍으로 이루어져 있습니다. 스윕 구성에서 최상위 키를 사용하여 검색할 파라미터([`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 키), 파라미터 공간 검색 방법([`method`]({{< relref path="./sweep-config-keys.md#method" lang="ko" >}}) 키) 등과 같은 스윕 검색의 특성을 정의할 수 있습니다.

아래 표는 스윕 구성에서 사용할 수 있는 최상위 키와 간략한 설명입니다. 각 키에 대한 자세한 정보는 해당 섹션을 참고하세요.

| 최상위 키      | 설명 |
| -------------- | ----------- |
| `program` | (필수) 실행할 트레이닝 스크립트 |
| `entity` | 이 스윕의 Entity |
| `project` | 이 스윕의 Project |
| `description` | 스윕에 대한 설명 텍스트 |
| `name` | 스윕의 이름이며, W&B UI에 표시됩니다. |
| [`method`]({{< relref path="#method" lang="ko" >}}) | (필수) 파라미터 검색 전략 |
| [`metric`]({{< relref path="#metric" lang="ko" >}}) | 최적화할 메트릭 (특정 검색 전략, 중지 기준에서만 사용) |
| [`parameters`]({{< relref path="#parameters" lang="ko" >}}) | (필수) 검색할 파라미터의 범위 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ko" >}}) | 얼리 스톱 기준 |
| [`command`]({{< relref path="#command" lang="ko" >}}) | 트레이닝 스크립트 호출 및 인수 전달용 커맨드 구조 |
| `run_cap` | 해당 스윕에서 실행될 최대 run 개수 |

스윕 구성 구조에 대한 자세한 정보는 [Sweep configuration]({{< relref path="./sweep-config-keys.md" lang="ko" >}}) 문서를 참고하세요.

## `metric`

최적화할 메트릭의 이름, 목표, 대상 메트릭을 지정하려면 `metric` 최상위 키를 사용합니다.

|키 | 설명 |
| -------- | --------------------------------------------------------- |
| `name`   | 최적화할 메트릭의 이름                                  |
| `goal`   | `minimize` 또는 `maximize`(기본값은 `minimize`)        |
| `target` | 최적화할 메트릭의 목표 값. run 이 지정한 target 값에 도달해도 새로운 run 을 만들지 않습니다. 이미 실행 중인 run 이 target 에 도달하면 run 이 완료될 때까지 액티브 에이전트가 대기하다가, 이후 에이전트가 새로운 run 생성을 중지합니다. |

## `parameters`
YAML 파일 또는 Python 스크립트에서, `parameters`를 최상위 키로 지정하세요. `parameters` 키 아래에, 최적화 대상 하이퍼파라미터 이름을 입력합니다. 대표적인 하이퍼파라미터로는 learning rate, 배치 크기, 에포크 수, 옵티마이저 등이 있고, 더 많습니다. 각 하이퍼파라미터마다, 하나 이상의 검색 제약 조건을 지정하세요.

아래 표는 지원되는 하이퍼파라미터 검색 제약 조건입니다. 하이퍼파라미터와 유스 케이스에 따라, 아래 중 하나를 골라 스윕 에이전트가 검색할 영역(분포일 경우) 또는 사용할 값(`value`, `values` 등)을 지정할 수 있습니다.

| 검색 제약 조건 | 설명 |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | 해당 하이퍼파라미터의 사용 가능한 모든 값 지정. `grid`와 호환됨.    |
| `value`         | 해당 하이퍼파라미터의 단일 값 지정. `grid`와 호환됨.  |
| `distribution`  | [분포]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ko" >}})를 지정하세요. 기본값에 대한 설명은 표 아래 참고. |
| `probabilities` | `random` 사용 시 `values` 각각이 선택될 확률 지정.  |
| `min`, `max`    | (`int` 또는 `float`) 최대 및 최소값. `int`이면 `int_uniform`, `float`이면 `uniform` 분포용 하이퍼파라미터. |
| `mu`            | (`float`) `normal` 또는 `lognormal` 분포 하이퍼파라미터용 평균값. |
| `sigma`         | (`float`) `normal` 또는 `lognormal` 분포 하이퍼파라미터용 표준편차. |
| `q`             | (`float`) 양자화된 하이퍼파라미터에 대한 스텝 크기.     |
| `parameters`    | 최상위 파라미터 하위에 다른 파라미터를 중첩하여 지정.   |

{{% alert %}}
[분포]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ko" >}})가 지정되지 않은 경우, W&B는 아래 조건에 따라 분포를 자동 지정합니다:
* `values` 지정 시 `categorical`
* `max`, `min`을 정수로 지정 시 `int_uniform`
* `max`, `min`을 실수로 지정 시 `uniform`
* `value`로 값을 지정 시 `constant`
{{% /alert %}}

## `method`
`method` 키로 하이퍼파라미터 검색 전략을 지정하세요. 총 세 가지 하이퍼파라미터 검색 전략이 있습니다: grid, random, Bayesian search.

#### Grid search
모든 하이퍼파라미터 값의 조합을 반복적으로 시도합니다. grid search는 각 반복에서 사용할 하이퍼파라미터 값을 별도의 정보 없이 조합하여 결정합니다. 연산량이 많아질 수 있습니다.     

연속적인 검색 공간을 사용하는 경우 grid search는 무한 반복됩니다.

#### Random search
분포를 기반으로 매 반복마다 무작위로 하이퍼파라미터 값을 선택합니다. random search는 커맨드라인, python 스크립트, 또는 [W&B 앱]({{< relref path="../sweeps-ui.md" lang="ko" >}})에서 프로세스를 직접 중단하지 않는다면 계속 실행됩니다.

`random`(`method: random`) 검색 시에는 metric 키로 분포 공간을 지정하세요.

#### Bayesian search
[random]({{< relref path="#random-search" lang="ko" >}}) 및 [grid]({{< relref path="#grid-search" lang="ko" >}}) 검색과 달리, Bayesian 모델은 데이터를 바탕으로 결정을 내립니다. 베이지안 최적화는 확률적 모델을 이용해서 대리 함수(surrogate function)에서 여러 값들을 반복적으로 테스트해보고 이후 목적 함수(objective function)를 평가하며, 사용할 값을 결정합니다. 베이지안 탐색은 연속 파라미터가 적을 때 효과적이지만, 파라미터가 많아질수록 확장성이 떨어집니다. 자세한 내용은 [Bayesian Optimization Primer 논문](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)을 참고하세요.

베이지안 탐색 역시 커맨드라인, 파이썬 스크립트, 또는 [W&B 앱]({{< relref path="../sweeps-ui.md" lang="ko" >}})에서 직접 중단하지 않는 한 계속 실행됩니다.

### Random 및 Bayesian search의 분포 옵션
`parameter` 키 내부에, 최적화할 하이퍼파라미터의 이름을 중첩한 뒤 `distribution` 키로 값의 분포를 지정합니다.

아래 표는 W&B에서 지원하는 분포 옵션입니다.

| `distribution` 키 값  | 설명            |
| ------------------------ | ------------------------------------ |
| `constant`               | 상수 분포. 사용할 상수 값(`value`)을 지정해야 함.                    |
| `categorical`            | 범주형 분포. 하이퍼파라미터의 모든 유효 값(`values`)을 지정해야 함. |
| `int_uniform`            | 정수형 균등 분포. `max`, `min`을 정수로 지정해야 함.     |
| `uniform`                | 연속형 균등 분포. `max`, `min`을 실수로 지정해야 함.      |
| `q_uniform`              | 양자화 균등 분포. `round(X / q) * q`를 반환(X는 uniform). `q` 기본값은 `1`.|
| `log_uniform`            | 로그-균등 분포. 자연로그가 `min`과 `max` 사이에서 균등분포가 되도록 `exp(min)`과 `exp(max)` 사이의 값 `X`를 반환.   |
| `log_uniform_values`     | 로그-균등 분포. `log(X)`가 `log(min)`과 `log(max)` 사이에서 균등분포가 되도록 `min`과 `max` 사이의 값 `X` 반환.     |
| `q_log_uniform`          | 양자화 로그-균등 분포. `round(X / q) * q`(X는 `log_uniform`). q 기본값 `1`. |
| `q_log_uniform_values`   | 양자화 로그-균등 분포. `round(X / q) * q`(X는 `log_uniform_values`). q 기본값 `1`.  |
| `inv_log_uniform`        | 역 로그-균등 분포. `log(1/X)`가 `min`과 `max` 사이에서 균등분포가 되도록 `X` 반환. |
| `inv_log_uniform_values` | 역 로그-균등 분포.  `log(1/X)`가 `log(1/max)`와 `log(1/min)` 사이에서 균등분포가 되도록 `X` 반환.    |
| `normal`                 | 정규 분포. 반환 값은 평균 `mu`(기본값 `0`), 표준편차 `sigma`(기본값 `1`)를 갖는 정규분포.|
| `q_normal`               | 양자화 정규분포. `round(X / q) * q`(X는 `normal`). Q 기본값 1.  |
| `log_normal`             | 로그 정규분포. 반환 값 `X`가 `log(X)`를 취하면 평균 `mu`(기본값 0), 표준편차 `sigma`(기본값 1)의 정규분포가 됨. |
| `q_log_normal`  | 양자화 로그 정규분포. `round(X / q) * q`(X는 `log_normal`). `q` 기본값 1. |

## `early_terminate`

`early_terminate` 기능을 사용하여 성능이 낮은 run 을 조기에 종료할 수 있습니다. 조기 종료가 발생하면, W&B는 현재 run 을 중단하고 다음 하이퍼파라미터 값으로 새 run 을 생성합니다.

{{% alert %}}
`early_terminate`를 사용할 경우 중지 알고리즘을 반드시 지정해야 합니다. 스윕 구성 내 `early_terminate` 하위에 `type` 키를 중첩하세요.
{{% /alert %}}

### Stopping algorithm

{{% alert %}}
현재 W&B에서는 [Hyperband](https://arxiv.org/abs/1603.06560) stopping algorithm만 지원합니다. 
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560)는 미리 정의된 반복 횟수(*brackets*)에서 각 프로그램을 중지할지 계속할지 평가하는 하이퍼파라미터 최적화 방식입니다.

W&B run 이 브래킷에 도달하면, 해당 run 의 metric 값과 지금까지 보고된 metric 값들을 비교합니다. 만약 목표가 최소화라면 metric 값이 너무 높을 때, 최대화라면 값이 너무 낮을 때 run 을 종료합니다.

브래킷은 로그된 반복 횟수를 기반으로 합니다. 브래킷 개수는 최적화 중인 metric 을 로그하는 횟수와 같습니다. 반복은 step, epoch, 또는 그 중간 값이 될 수 있습니다. step 카운터의 실제 값은 브래킷 계산에 사용되지 않습니다.

{{% alert %}}
브래킷 스케줄을 생성하려면 `min_iter` 또는 `max_iter`를 지정하세요.
{{% /alert %}}

| 키         | 설명                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 첫 번째 브래킷의 반복 횟수 지정                    |
| `max_iter` | 반복 횟수의 최대값 지정                     |
| `s`        | 브래킷의 총 개수 지정 (`max_iter` 사용 시 필수) |
| `eta`      | 브래킷 곱셈 스케줄 지정 (기본값: `3`)        |
| `strict`   | 'strict' 모드로 동작하여 Hyperband 원 논문을 더 엄격하게 따르며, run 을 적극적으로 pruning. 기본값은 false. |

{{% alert %}}
[run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})을 종료할지 Hyperband가 몇 분마다 체크합니다. run 또는 한 번의 반복이 짧으면 실제 종료 시점이 브래킷에서 지정된 시점과 다를 수 있습니다.
{{% /alert %}}

## `command` 

`command` 키 내에 중첩 값을 추가하여 포맷 및 구성 요소를 세밀하게 조절할 수 있습니다. 파일명 등 고정된 요소를 직접 포함할 수 있습니다.

{{% alert %}}
유닉스 시스템에서 `/usr/bin/env`는 환경에 따라 올바른 파이썬 인터프리터가 선택되도록 보장해줍니다.
{{% /alert %}}

W&B는 커맨드 내 가변 요소를 위한 아래 매크로를 지원합니다:

| 커맨드 매크로               | 설명                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | 유닉스 시스템에서는 `/usr/bin/env`, 윈도우에서는 제외됨.                                                                                                                   |
| `${interpreter}`           | `python`으로 확장됨.                                                                                                                                                  |
| `${program}`               | 스윕 구성 `program` 키에 지정한 트레이닝 스크립트 파일명으로 치환.                                                                                          |
| `${args}`                  | 하이퍼파라미터 및 값을 `--param1=value1 --param2=value2` 형식으로 입력.                                                                                 |
| `${args_no_boolean_flags}` | 불리언 파라미터의 경우 `--boolean_flag_param` (True일 때) 또는 누락(False일 때) 방식, 그 외는 `--param1=value1` 형식.                           |
| `${args_no_hyphens}`       | 하이퍼파라미터 및 값을 `param1=value1 param2=value2` 형식으로 입력.                                                                                           |
| `${args_json}`             | 하이퍼파라미터 및 값을 JSON 인코딩으로 입력.                                                                                                                     |
| `${args_json_file}`        | 하이퍼파라미터 및 값을 JSON으로 인코딩한 파일 경로 입력.                                                                                   |
| `${envvar}`                | 환경 변수 전달용. `${envvar:MYENVVAR}` __는 MYENVVAR 환경 변수의 값으로 확장됩니다.__                                               |