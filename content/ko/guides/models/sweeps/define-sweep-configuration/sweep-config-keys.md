---
title: Sweep configuration options
menu:
  default:
    identifier: ko-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

스윕 구성은 중첩된 키-값 쌍으로 구성됩니다. 스윕 구성 내에서 최상위 키를 사용하여 검색할 파라미터 ( [`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ko" >}}) 키), 파라미터 공간을 검색하는 방법 ( [`method`]({{< relref path="./sweep-config-keys.md#method" lang="ko" >}}) 키) 등과 같은 스윕 검색의 특성을 정의합니다.

다음 표는 최상위 스윕 구성 키와 간단한 설명을 나열합니다. 각 키에 대한 자세한 내용은 해당 섹션을 참조하십시오.

| 최상위 키 | 설명 |
| -------------- | ----------- |
| `program` | (필수) 실행할 트레이닝 스크립트 |
| `entity` | 이 스윕의 엔티티 |
| `project` | 이 스윕의 프로젝트 |
| `description` | 스윕에 대한 텍스트 설명 |
| `name` | W&B UI에 표시되는 스윕 이름 |
| [`method`]({{< relref path="#method" lang="ko" >}}) | (필수) 검색 전략 |
| [`metric`]({{< relref path="#metric" lang="ko" >}}) | 최적화할 메트릭 (특정 검색 전략 및 중지 기준에만 사용됨) |
| [`parameters`]({{< relref path="#parameters" lang="ko" >}}) | (필수) 검색할 파라미터 범위 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ko" >}}) | 조기 중단 기준 |
| [`command`]({{< relref path="#command" lang="ko" >}}) | 트레이닝 스크립트를 호출하고 인수를 전달하기 위한 코맨드 구조 |
| `run_cap` | 이 스윕의 최대 Runs 수 |

스윕 구성 구조에 대한 자세한 내용은 [스윕 구성]({{< relref path="./sweep-config-keys.md" lang="ko" >}})을 참조하십시오.

## `metric`

`metric` 최상위 스윕 구성 키를 사용하여 최적화할 이름, 목표 및 대상 메트릭을 지정합니다.

| 키 | 설명 |
| -------- | --------------------------------------------------------- |
| `name` | 최적화할 메트릭 이름입니다. |
| `goal` | `minimize` 또는 `maximize` (기본값은 `minimize`). |
| `target` | 최적화하려는 메트릭의 목표 값입니다. 스윕은 사용자가 지정한 목표 값에 도달하면 새 Runs을 생성하지 않습니다. Run이 실행 중인 활성 에이전트는 (Run이 목표에 도달하면) 에이전트가 새 Runs 생성을 중지하기 전에 Run이 완료될 때까지 기다립니다. |

## `parameters`
YAML 파일 또는 Python 스크립트에서 `parameters`를 최상위 키로 지정합니다. `parameters` 키 내에서 최적화하려는 하이퍼파라미터의 이름을 제공합니다. 일반적인 하이퍼파라미터에는 학습률, 배치 크기, 에포크, 옵티마이저 등이 있습니다. 스윕 구성에서 정의하는 각 하이퍼파라미터에 대해 하나 이상의 검색 제약 조건을 지정합니다.

다음 표는 지원되는 하이퍼파라미터 검색 제약 조건을 보여줍니다. 하이퍼파라미터 및 유스 케이스에 따라 아래 검색 제약 조건 중 하나를 사용하여 스윕 에이전트에게 검색하거나 사용할 위치 (분포의 경우) 또는 내용 (`value`, `values` 등)을 알려줍니다.

| 검색 제약 조건 | 설명 |
| --------------- | ------------------------------------------------------------------------------ |
| `values` | 이 하이퍼파라미터에 대한 모든 유효한 값을 지정합니다. `grid`와 호환됩니다. |
| `value` | 이 하이퍼파라미터에 대한 단일 유효한 값을 지정합니다. `grid`와 호환됩니다. |
| `distribution` | 확률 [분포]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ko" >}})를 지정합니다. 기본값에 대한 정보는 이 표 다음의 참고 사항을 참조하십시오. |
| `probabilities` | `random`을 사용할 때 `values`의 각 요소를 선택할 확률을 지정합니다. |
| `min`, `max` | (`int` 또는 `float`) 최대값 및 최소값. `int`인 경우 `int_uniform` 분포 하이퍼파라미터의 경우. `float`인 경우 `uniform` 분포 하이퍼파라미터의 경우. |
| `mu` | (`float`) `normal` 또는 `lognormal` 분포 하이퍼파라미터의 평균 파라미터입니다. |
| `sigma` | (`float`) `normal` 또는 `lognormal` 분포 하이퍼파라미터의 표준 편차 파라미터입니다. |
| `q` | (`float`) 양자화된 하이퍼파라미터의 양자화 단계 크기입니다. |
| `parameters` | 루트 수준 파라미터 내부에 다른 파라미터를 중첩합니다. |

{{% alert %}}
W&B는 [분포]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ko" >}})가 지정되지 않은 경우 다음 조건에 따라 다음 분포를 설정합니다.
* `values`를 지정하면 `categorical`
* `max` 및 `min`을 정수로 지정하면 `int_uniform`
* `max` 및 `min`을 float로 지정하면 `uniform`
* `value`로 설정하면 `constant`
{{% /alert %}}

## `method`
`method` 키로 하이퍼파라미터 검색 전략을 지정합니다. 선택할 수 있는 세 가지 하이퍼파라미터 검색 전략이 있습니다: 그리드, 랜덤, 베이지안 탐색.
#### 그리드 탐색
하이퍼파라미터 값의 모든 조합을 반복합니다. 그리드 탐색은 각 반복에서 사용할 하이퍼파라미터 값 세트에 대해 정보에 입각하지 않은 결정을 내립니다. 그리드 탐색은 계산 비용이 많이 들 수 있습니다.

그리드 탐색은 연속 검색 공간 내에서 검색하는 경우 영원히 실행됩니다.

#### 랜덤 검색
각 반복에서 분포를 기반으로 무작위의 정보에 입각하지 않은 하이퍼파라미터 값 세트를 선택합니다. 커맨드라인, Python 스크립트 또는 [W&B 앱 UI]({{< relref path="../sweeps-ui.md" lang="ko" >}})에서 프로세스를 중지하지 않으면 랜덤 검색은 영원히 실행됩니다.

랜덤 (`method: random`) 검색을 선택하는 경우 메트릭 키로 분포 공간을 지정합니다.

#### 베이지안 탐색
[랜덤]({{< relref path="#random-search" lang="ko" >}}) 및 [그리드]({{< relref path="#grid-search" lang="ko" >}}) 검색과 달리 베이지안 모델은 정보에 입각한 결정을 내립니다. 베이지안 최적화는 확률 모델을 사용하여 목적 함수를 평가하기 전에 대리 함수에서 값을 테스트하는 반복적인 프로세스를 통해 사용할 값을 결정합니다. 베이지안 탐색은 적은 수의 연속 파라미터에 적합하지만 확장성이 떨어집니다. 베이지안 탐색에 대한 자세한 내용은 [베이지안 최적화 입문서](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)를 참조하십시오.

커맨드라인, Python 스크립트 또는 [W&B 앱 UI]({{< relref path="../sweeps-ui.md" lang="ko" >}})에서 프로세스를 중지하지 않으면 베이지안 탐색은 영원히 실행됩니다.

### 랜덤 및 베이지안 탐색을 위한 분포 옵션
`parameter` 키 내에서 하이퍼파라미터의 이름을 중첩합니다. 다음으로 `distribution` 키를 지정하고 값에 대한 분포를 지정합니다.

다음 표는 W&B가 지원하는 분포를 나열합니다.

| `distribution` 키 값 | 설명 |
| ------------------------ | ------------------------------------ |
| `constant` | 상수 분포. 사용할 상수 값 (`value`)을 지정해야 합니다. |
| `categorical` | 범주형 분포. 이 하이퍼파라미터에 대한 모든 유효한 값 (`values`)을 지정해야 합니다. |
| `int_uniform` | 정수에 대한 이산 균등 분포. `max` 및 `min`을 정수로 지정해야 합니다. |
| `uniform` | 연속 균등 분포. `max` 및 `min`을 float로 지정해야 합니다. |
| `q_uniform` | 양자화된 균등 분포. X가 균등인 `round(X / q) * q`를 반환합니다. `q`의 기본값은 `1`입니다. |
| `log_uniform` | 로그-균등 분포. 자연 로그가 `min`과 `max` 사이에서 균등하게 분포되도록 `exp(min)`과 `exp(max)` 사이의 값 `X`를 반환합니다. |
| `log_uniform_values` | 로그-균등 분포. `log(X)`가 `log(min)`과 `log(max)` 사이에서 균등하게 분포되도록 `min`과 `max` 사이의 값 `X`를 반환합니다. |
| `q_log_uniform` | 양자화된 로그 균등. `X`가 `log_uniform`인 `round(X / q) * q`를 반환합니다. `q`의 기본값은 `1`입니다. |
| `q_log_uniform_values` | 양자화된 로그 균등. `X`가 `log_uniform_values`인 `round(X / q) * q`를 반환합니다. `q`의 기본값은 `1`입니다. |
| `inv_log_uniform` | 역 로그 균등 분포. `log(1/X)`가 `min`과 `max` 사이에서 균등하게 분포되는 `X`를 반환합니다. |
| `inv_log_uniform_values` | 역 로그 균등 분포. `log(1/X)`가 `log(1/max)`와 `log(1/min)` 사이에서 균등하게 분포되는 `X`를 반환합니다. |
| `normal` | 정규 분포. 반환 값은 평균 `mu` (기본값 `0`) 및 표준 편차 `sigma` (기본값 `1`)로 정규 분포됩니다. |
| `q_normal` | 양자화된 정규 분포. `X`가 `normal`인 `round(X / q) * q`를 반환합니다. Q의 기본값은 1입니다. |
| `log_normal` | 로그 정규 분포. 자연 로그 `log(X)`가 평균 `mu` (기본값 `0`) 및 표준 편차 `sigma` (기본값 `1`)로 정규 분포되도록 값 `X`를 반환합니다. |
| `q_log_normal` | 양자화된 로그 정규 분포. `X`가 `log_normal`인 `round(X / q) * q`를 반환합니다. `q`의 기본값은 `1`입니다. |

## `early_terminate`

조기 종료 (`early_terminate`)를 사용하여 성능이 저조한 Runs을 중지합니다. 조기 종료가 발생하면 W&B는 새로운 하이퍼파라미터 값 세트로 새로운 Run을 만들기 전에 현재 Run을 중지합니다.

{{% alert %}}
`early_terminate`를 사용하는 경우 중지 알고리즘을 지정해야 합니다. 스윕 구성 내에서 `early_terminate` 내에 `type` 키를 중첩합니다.
{{% /alert %}}

### 중지 알고리즘

{{% alert %}}
W&B는 현재 [Hyperband](https://arxiv.org/abs/1603.06560) 중지 알고리즘을 지원합니다.
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) 하이퍼파라미터 최적화는 프로그램을 중지해야 하는지 또는 미리 설정된 하나 이상의 반복 횟수 ( *브래킷* 이라고 함)에서 계속해야 하는지 평가합니다.

W&B Run이 브래킷에 도달하면 스윕은 해당 Run의 메트릭을 이전에 보고된 모든 메트릭 값과 비교합니다. Run의 메트릭 값이 너무 높으면 (목표가 최소화인 경우) 또는 Run의 메트릭이 너무 낮으면 (목표가 최대화인 경우) 스윕은 Run을 종료합니다.

브래킷은 기록된 반복 횟수를 기반으로 합니다. 브래킷 수는 최적화하는 메트릭을 기록하는 횟수에 해당합니다. 반복은 단계, 에포크 또는 그 사이의 어떤 것에 해당할 수 있습니다. 단계 카운터의 숫자 값은 브래킷 계산에 사용되지 않습니다.

{{% alert %}}
브래킷 일정을 만들려면 `min_iter` 또는 `max_iter`를 지정합니다.
{{% /alert %}}

| 키 | 설명 |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 첫 번째 브래킷의 반복을 지정합니다. |
| `max_iter` | 최대 반복 횟수를 지정합니다. |
| `s` | 총 브래킷 수를 지정합니다 (`max_iter`에 필요). |
| `eta` | 브래킷 승수 일정을 지정합니다 (기본값: `3`). |
| `strict`   | 원본 Hyperband 논문을 더 밀접하게 따르는 공격적으로 Runs을 정리하는 'strict' 모드를 활성화합니다. 기본값은 false입니다. |

{{% alert %}}
Hyperband는 몇 분마다 종료할 [W&B Runs]({{< relref path="/ref/python/run.md" lang="ko" >}})을 확인합니다. Run 또는 반복이 짧으면 종료 Run 타임스탬프가 지정된 브래킷과 다를 수 있습니다.
{{% /alert %}}

## `command`

`command` 키 내에 중첩된 값으로 형식과 내용을 수정합니다. 파일 이름과 같은 고정된 구성 요소를 직접 포함할 수 있습니다.

{{% alert %}}
Unix 시스템에서 `/usr/bin/env`는 OS가 환경에 따라 올바른 Python 인터프리터를 선택하도록 합니다.
{{% /alert %}}

W&B는 코맨드의 가변 구성 요소에 대해 다음 매크로를 지원합니다.

| 코맨드 매크로 | 설명 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}` | Unix 시스템의 경우 `/usr/bin/env`, Windows의 경우 생략됩니다. |
| `${interpreter}` | `python`으로 확장됩니다. |
| `${program}` | 스윕 구성 `program` 키로 지정된 트레이닝 스크립트 파일 이름입니다. |
| `${args}` | `--param1=value1 --param2=value2` 형식의 하이퍼파라미터 및 해당 값입니다. |
| `${args_no_boolean_flags}` | `--param1=value1` 형식의 하이퍼파라미터 및 해당 값입니다. 단, 부울 파라미터는 `True`인 경우 `--boolean_flag_param` 형식이고 `False`인 경우 생략됩니다. |
| `${args_no_hyphens}` | `param1=value1 param2=value2` 형식의 하이퍼파라미터 및 해당 값입니다. |
| `${args_json}` | JSON으로 인코딩된 하이퍼파라미터 및 해당 값입니다. |
| `${args_json_file}` | JSON으로 인코딩된 하이퍼파라미터 및 해당 값을 포함하는 파일의 경로입니다. |
| `${envvar}` | 환경 변수를 전달하는 방법입니다. `${envvar:MYENVVAR}` __은 MYENVVAR 환경 변수의 값으로 확장됩니다. __ |