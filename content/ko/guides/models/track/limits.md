---
title: Experiments 제한 및 성능
description: W&B에서 페이지를 더 빠르고 반응성 있게 유지하려면 아래 제안된 범위 내에서 로그를 진행하세요.
menu:
  default:
    identifier: ko-guides-models-track-limits
    parent: experiments
weight: 7
---

W&B에서 페이지를 더 빠르고 반응성 있게 유지하려면 다음의 권장 범위 내에서 로그를 남기세요.

## 로깅 시 고려사항

`wandb.Run.log()`를 사용해 실험 메트릭을 추적하세요.

### 고유 메트릭 개수

더 빠른 성능을 위해 프로젝트 내 총 고유 메트릭 개수를 10,000개 미만으로 유지하세요.

```python
import wandb

with wandb.init() as run:
    run.log(
        {
            "a": 1,  # "a"는 고유 메트릭입니다
            "b": {
                "c": "hello",  # "b.c"는 고유 메트릭입니다
                "d": [1, 2, 3],  # "b.d"는 고유 메트릭입니다
            },
        }
    )
```

{{% alert %}}
W&B는 중첩된 값을 자동으로 펴서 저장합니다. 즉, 딕셔너리를 전달하면 W&B는 점(`.`)으로 구분된 이름으로 바꿉니다. config 값의 경우 이름에 최대 3개의 점까지, summary 값은 4개의 점까지 지원합니다.
{{% /alert %}}

워크스페이스가 갑자기 느려졌다면, 최근 runs에서 수천 개의 새로운 메트릭이 의도치 않게 로그되었는지 확인하세요. (이 문제는 수천 개의 plot이 있지만 실제로는 한두 개의 run만 표시되는 섹션에서 쉽게 찾을 수 있습니다.) 만약 그렇다면, 해당 run을 삭제하고 원하는 메트릭으로 다시 생성하는 것을 고려하세요.

### 값의 크기

단일로 로그되는 값의 크기는 1MB 미만, 한 번의 `run.log` 호출 당 총 크기는 25MB 미만으로 제한하세요. 이 제한은 `wandb.Image`, `wandb.Audio` 등과 같은 `wandb.Media` 타입에는 적용되지 않습니다.

```python
import wandb

run = wandb.init(project="wide-values")

# 권장하지 않음
run.log({"wide_key": range(10000000)})

# 권장하지 않음
with open("large_file.json", "r") as f:
    large_data = json.load(f)
    run.log(large_data)

run.finish()
```

너무 큰 값은 해당 run의 모든 메트릭 plot 로딩 속도에 영향을 줄 수 있습니다. 이는 해당 메트릭 뿐 아니라 전체 metric에 영향을 줍니다.

{{% alert %}}
권장 범위를 넘는 값을 로그해도 데이터는 저장되고 추적됩니다. 하지만 plot 로딩이 느려질 수 있습니다.
{{% /alert %}}

### 메트릭 로깅 빈도

로깅하는 메트릭에 맞는 적절한 빈도를 설정하세요. 일반적으로, 값이 큰 메트릭은 더 적게, 값이 작은 메트릭은 더 자주 로깅하는 것이 좋습니다. W&B에서는 다음을 권장합니다:

- 스칼라: 메트릭 당 100,000개 미만
- 미디어: 메트릭 당 50,000개 미만
- 히스토그램: 메트릭 당 10,000개 미만

```python
import wandb

with wandb.init(project="metric-frequency") as run:
    # 권장하지 않음
    run.log(
        {
            "scalar": 1,  # 100,000 스칼라
            "media": wandb.Image(...),  # 100,000 이미지
            "histogram": wandb.Histogram(...),  # 100,000 히스토그램
        }
    )

    # 권장함
    run.log(
        {
            "scalar": 1,  # 100,000 스칼라
        },
        commit=True,
    )  # 여러 스텝의 메트릭을 한 번에 커밋

    run.log(
        {
            "media": wandb.Image(...),  # 50,000 이미지
        },
        commit=False,
    )
    
    run.log(
        {
            "histogram": wandb.Histogram(...),  # 10,000 히스토그램
        },
        commit=False,
    )
```

{{% alert %}}
W&B는 데이터를 계속 받아들이지만 권장 범위를 초과하면 페이지 로딩이 느려질 수 있습니다.
{{% /alert %}}

### Config 크기

run config의 총 크기를 10MB 미만으로 제한하세요. 너무 큰 값을 기록하면 프로젝트 워크스페이스와 runs 테이블의 동작이 느려질 수 있습니다.

```python
import wandb 

# 권장함
with wandb.init(
    project="config-size",
    config={
        "lr": 0.1,
        "batch_size": 32,
        "epochs": 4,
    }
) as run:
    # 여기에 트레이닝 코드를 작성하세요
    pass

# 권장하지 않음
with wandb.init(
    project="config-size",
    config={
        "large_list": list(range(10000000)),  # 매우 큰 리스트
        "large_string": "a" * 10000000,  # 매우 긴 문자열
    }
) as run:
    # 여기에 트레이닝 코드를 작성하세요
    pass

# 권장하지 않음
with open("large_config.json", "r") as f:
    large_config = json.load(f)
    wandb.init(config=large_config)
```

## 워크스페이스 관련 고려사항

### Run 개수

로딩 시간을 줄이기 위해, 단일 프로젝트 내의 run 개수를 다음 범위 안에서 유지하세요:

- SaaS Cloud: 100,000 개
- 전용 클라우드 또는 셀프 매니지드: 10,000 개

이 한도를 초과하면 프로젝트 워크스페이스나 runs 테이블, 특히 run을 그룹화하거나 하나의 run에서 수많은 고유 메트릭을 모으는 경우 작업이 느려질 수 있습니다. 자세한 내용은 [메트릭 개수]({{< relref path="#metric-count" lang="ko" >}}) 섹션을 참고하세요.

팀이 동일한 runs 세트(예: 최근 run 세트)에 자주 엑세스한다면, [자주 사용하지 않는 runs를 한 번에 "아카이브" 프로젝트로 옮기고]({{< relref path="/guides/models/track/runs/manage-runs.md" lang="ko" >}}) 작업 프로젝트에는 소수의 run만 남겨두세요.

### 워크스페이스 성능
이 섹션에서는 워크스페이스 성능을 최적화하는 팁을 제공합니다.

#### 패널 개수
기본적으로 워크스페이스는 _자동_이고, 로그된 각 키에 대해 표준 패널을 자동으로 생성합니다. 큰 프로젝트의 워크스페이스에 많은 패널이 포함되면 워크스페이스 로딩 및 사용이 느려질 수 있습니다. 성능을 향상하려면 다음과 같이 하세요:

1. 워크스페이스를 수동 모드로 재설정하세요. 이 경우 기본적으로 패널이 포함되지 않습니다.
1. 시각화가 필요한 키만 [빠른 추가]({{< relref path="/guides/models/app/features/panels/#quick-add" lang="ko" >}})로 직접 패널을 선택해서 추가하세요.

{{% alert %}}
사용하지 않는 패널을 하나씩 삭제해도 성능에 큰 영향이 없습니다. 대신 워크스페이스를 리셋하고 필요한 패널만 선별적으로 추가하세요.
{{% /alert %}}

워크스페이스 구성에 대해 더 알고 싶다면 [패널]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})을 참고하세요.

#### 섹션 개수

워크스페이스에 수백 개의 섹션이 있으면 성능이 저하될 수 있습니다. 메트릭의 상위 그룹을 기준으로 섹션을 만들고, 하나의 메트릭마다 섹션을 만드는 방식은 피하세요.

섹션이 너무 많아 성능이 떨어진다면, 접두사별로 섹션을 생성하게 설정을 변경하는 것도 고려해 보세요. 이렇게 하면 섹션 수가 줄고 성능이 향상될 수 있습니다.

{{< img src="/images/track/section_prefix_toggle.gif" alt="Toggling section creation" >}}

### 메트릭 개수

run 당 5,000~100,000개의 메트릭을 기록한다면 [수동 워크스페이스]({{< relref path="/guides/models/app/features/panels/#workspace-modes" lang="ko" >}}) 사용을 권장합니다. Manual 모드에서는 다양한 메트릭 세트를 빠르게 추가하거나 여러 개를 한 번에 삭제할 수 있습니다. plot 수가 적으면 워크스페이스가 더 빠르게 로드됩니다. plot되지 않은 메트릭도 정상적으로 수집 및 저장됩니다.

워크스페이스를 수동 모드로 재설정하려면, 워크스페이스의 '...' 액션 메뉴를 클릭한 뒤 **Reset workspace**를 선택하세요. 워크스페이스를 리셋해도 run의 저장된 메트릭에는 영향이 없습니다. 자세한 내용은 [워크스페이스 패널 관리]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}})를 참고하세요.

### 파일 개수

단일 run에서 업로드하는 파일 개수는 1,000개 이하로 유지하세요. 많은 파일을 로그해야 한다면 W&B Artifacts 활용을 추천합니다. run 당 1,000개를 초과하는 파일을 등록하면 run 페이지의 로딩 속도가 느려질 수 있습니다.

### 리포트와 워크스페이스

리포트는 패널, 텍스트, 미디어 등을 자유롭게 배치해 동료들과 인사이트를 쉽게 공유할 수 있습니다.

반면, 워크스페이스는 수십~수천 개의 메트릭과 수백~수십만 개의 run도 빠르게 분석할 수 있도록 고밀도 및 고성능 분석 환경을 제공합니다. 워크스페이스는 리포트에 비해 캐싱, 쿼리, 로딩이 더 최적화되어 있습니다. 따라서 분석이 주 목적이거나 20개 이상의 plot을 한 번에 표시해야 하는 프로젝트라면 워크스페이스 사용을 추천합니다.

## Python 스크립트 성능

python 스크립트 성능이 떨어지는 주요 원인은 다음과 같습니다:

1. 데이터가 너무 큽니다. 대용량 데이터는 트레이닝 루프에 1ms 이상의 오버헤드를 가져올 수 있습니다.
2. 네트워크 속도와 W&B 백엔드 구성 방식입니다.
3. `wandb.Run.log()`를 초당 여러 번 호출하는 경우. 호출할 때마다 트레이닝 루프에 소량의 지연이 추가됩니다.

{{% alert %}}
자주 로깅을 해서 트레이닝 run 속도가 느려졌나요? 더 나은 로깅 전략으로 성능을 개선하는 방법은 [이 Colab](https://wandb.me/log-hf-colab)을 참고하세요.
{{% /alert %}}

W&B는 속도 제한(rate limiting) 외에 별도의 제한을 두지 않습니다. W&B Python SDK는 제한을 초과한 요청에 대해 자동으로 지수 백오프와 재시도(backoff & retry)를 수행합니다. W&B Python SDK는 커맨드라인에서 “Network failure” 메시지로 응답합니다. 무료 계정에서 합리적인 사용 범위를 심각하게 초과하는 경우, W&B가 따로 연락할 수 있습니다.

## 속도 제한(Rate limits)

W&B SaaS Cloud API는 시스템 안정성과 가용성을 위해 속도 제한(rate limit)을 적용합니다. 이는 공유 인프라에서 특정 사용자가 리소스를 독점적으로 사용하는 것을 방지하고, 모든 사용자에게 서비스가 원활히 제공되게 합니다. 속도 제한이 더 낮아질 수 있는 여러 상황이 있습니다.

{{% alert %}}
속도 제한은 변경될 수 있습니다.
{{% /alert %}}

속도 제한에 걸리면 HTTP `429` `Rate limit exceeded` 오류와 함께, [rate limit HTTP 헤더]({{< relref path="#rate-limit-http-headers" lang="ko" >}})가 응답에 포함됩니다.

### rate limit HTTP 헤더

아래 표는 rate limit HTTP 헤더에 대해 설명합니다:

| 헤더 이름                | 설명                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------ |
| RateLimit-Limit     | 시간 단위로 허용되는 전체 쿼터(0~1000 범위)                                               |
| RateLimit-Remaining | 현재 rate limit 윈도우 내 남은 쿼터(0~1000 범위)                                         |
| RateLimit-Reset     | 현재 쿼터가 재설정될 때까지 남은 초                                                      |

### 메트릭 로깅 API의 속도 제한

`wandb.Run.log()`는 트레이닝 데이터를 W&B에 기록합니다. 이 API는 온라인 혹은 [오프라인 동기화]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 방식으로 사용할 수 있습니다. 두 경우 모두, 슬라이딩 타임 윈도우(rolling time window)마다 요청 크기 및 요청 빈도에 제한이 있습니다.

W&B는 W&B 프로젝트별로 rate limit을 적용합니다. 즉, 한 팀에 3개 프로젝트가 있다면, 각각 독립적으로 할당량이 적용됩니다. [유료 플랜](https://wandb.ai/site/pricing) 사용자는 무료 플랜 대비 더 높은 제한을 가집니다.

속도 제한에 걸리면 HTTP `429` `Rate limit exceeded` 오류와 함께, [rate limit HTTP 헤더]({{< relref path="#rate-limit-http-headers" lang="ko" >}})가 응답에 포함됩니다.

### 메트릭 로깅 API 속도 제한 준수 팁

속도 제한을 초과하면 `run.finish()`가 rate limit이 해제될 때까지 지연될 수 있습니다. 이를 피하려면, 다음 전략을 고려하세요:

- W&B Python SDK 버전 업데이트: 최신 버전의 W&B Python SDK를 사용하세요. 최신 버전은 쿼터 사용 최적화 및 자동 재시도 등의 기능이 강화되어 있습니다.
- 메트릭 로깅 주기 줄이기:
  쿼터 사용을 아끼기 위해 로그 빈도를 줄이세요. 예를 들어, 모든 에포크마다 로그하는 대신 5 에포크마다 기록하도록 변경할 수 있습니다.

```python
import wandb
import random

with wandb.init(project="basic-intro") as run:
    for epoch in range(10):
        # 트레이닝 및 평가 시뮬레이션
        accuracy = 1 - 2 ** -epoch - random.random() / epoch
        loss = 2 ** -epoch + random.random() / epoch

        # 5 에포크마다 메트릭을 기록
        if epoch % 5 == 0:
            run.log({"acc": accuracy, "loss": loss})
```

- 수동 데이터 동기화: 만약 rate limit에 걸리면 run 데이터가 로컬에 저장됩니다. 이 데이터를 `wandb sync <run-file-path>` 커맨드로 직접 동기화할 수 있습니다. 자세한 내용은 [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ko" >}}) 참고.

### GraphQL API의 속도 제한

W&B Models UI와 SDK의 [public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})는 데이터를 쿼리/수정하기 위해 서버에 GraphQL 요청을 보냅니다. SaaS Cloud에서, 모든 GraphQL 요청은 비로그인 요청 시 IP주소 기준, 로그인 요청 시 사용자 기준으로 rate limit이 적용됩니다. 제한은 일정 시간 내 초당 요청 횟수에 따라 결정되며, 요금제에 따라 기본값이 다릅니다. 프로젝트 경로가 명시된(예: reports, runs, artifacts) SDK 요청은 프로젝트별로, 요청 시 데이터베이스 쿼리 시간 기준으로 제한됩니다.

[Teams 및 Enterprise 플랜](https://wandb.ai/site/pricing) 회원은 무료 플랜보다 더 높은 제한이 적용됩니다.
W&B Models SDK의 public API 사용 중 rate limit을 초과하면, 표준 출력에 관련 오류 메시지가 나타납니다.

속도 제한에 걸리면 HTTP `429` `Rate limit exceeded` 오류와 함께, [rate limit HTTP 헤더]({{< relref path="#rate-limit-http-headers" lang="ko" >}})가 응답에 포함됩니다.

#### GraphQL API 속도 제한 준수 팁

W&B Models SDK의 [public API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})로 대량의 데이터를 가져오는 경우, 요청 사이에 최소 1초의 대기 시간을 두세요. 만약 HTTP `429` `Rate limit exceeded` 오류나, 응답 헤더에서 `RateLimit-Remaining=0`을 받았다면, `RateLimit-Reset`에 표시된 초만큼 기다렸다가 재시도하세요.

## 브라우저 관련 팁

W&B 앱은 메모리 소모가 큰 서비스이므로 Chrome에서 최고 성능을 발휘합니다. 컴퓨터의 메모리에 따라 W&B을 3개 이상의 탭에서 동시에 실행하면 성능 저하가 생길 수 있습니다. 예기치 않게 느려질 경우, 다른 탭이나 애플리케이션을 닫아보세요.

## 성능 문제 W&B에 보고하기

W&B는 성능 문제를 매우 중요하게 생각하며 모든 지연 신고를 조사합니다. 빠른 조치를 위해, 느린 로딩 현상을 보고할 때 주요 메트릭과 이벤트를 캡처하는 W&B 내장 퍼포먼스 로거를 활용할 수 있습니다. 로딩이 느린 페이지 URL 끝에 `&PERF_LOGGING`을 추가하고, 콘솔 출력 결과를 담당 계정팀이나 Support에 전달해 주세요.

{{< img src="/images/track/adding_perf_logging.gif" alt="Adding PERF_LOGGING" >}}