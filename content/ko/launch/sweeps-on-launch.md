---
title: Create sweeps with W&B Launch
description: Launch에서 하이퍼파라미터 스윕을 자동화하는 방법을 알아보세요.
menu:
  launch:
    identifier: ko-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B **Launch**를 사용하여 하이퍼파라미터 튜닝 작업( [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}))을 생성합니다. **Launch**에서 **Sweeps**를 사용하면 스윕 스케줄러가 스윕할 지정된 하이퍼파라미터와 함께 **Launch** 대기열로 푸시됩니다. 스윕 에이전트가 스윕 스케줄러를 선택하면 스윕 스케줄러가 시작되어 선택한 하이퍼파라미터로 스윕 **Run**을 동일한 대기열로 실행합니다. 이 프로세스는 스윕이 완료되거나 중지될 때까지 계속됩니다.

기본 W&B **Sweep** 스케줄링 엔진을 사용하거나 사용자 정의 스케줄러를 구현할 수 있습니다.

1. 표준 스윕 스케줄러: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 제어하는 기본 W&B **Sweep** 스케줄링 엔진을 사용합니다. 익숙한 `bayes`, `grid` 및 `random` 메소드를 사용할 수 있습니다.
2. 사용자 정의 스윕 스케줄러: 스윕 스케줄러가 작업으로 실행되도록 구성합니다. 이 옵션을 사용하면 완전히 사용자 정의할 수 있습니다. 표준 스윕 스케줄러를 확장하여 로깅을 더 많이 포함하는 방법의 예는 아래 섹션에서 확인할 수 있습니다.
 
{{% alert %}}
이 가이드에서는 W&B **Launch**가 이미 구성되었다고 가정합니다. W&B **Launch**가 구성되지 않은 경우 **Launch** 문서의 [시작 방법]({{< relref path="./#how-to-get-started" lang="ko" >}}) 섹션을 참조하십시오.
{{% /alert %}}

{{% alert %}}
**Launch**에서 **Sweeps**를 처음 사용하는 경우 '기본' 메소드를 사용하여 **Launch**에서 스윕을 생성하는 것이 좋습니다. 표준 W&B 스케줄링 엔진이 요구 사항을 충족하지 못하는 경우 사용자 정의 **Launch** 스윕 스케줄러를 사용하십시오.
{{% /alert %}}

## W&B 표준 스케줄러로 스윕 생성하기
**Launch**를 사용하여 W&B **Sweeps**를 생성합니다. W&B 앱을 사용하여 대화식으로 스윕을 생성하거나 W&B CLI를 사용하여 프로그래밍 방식으로 스윕을 생성할 수 있습니다. 스케줄러를 사용자 정의하는 기능을 포함하여 **Launch** 스윕의 고급 구성의 경우 CLI를 사용하십시오.

{{% alert %}}
W&B **Launch**로 스윕을 생성하기 전에 먼저 스윕할 작업을 생성해야 합니다. 자세한 내용은 [작업 생성]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ko" >}}) 페이지를 참조하십시오.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B 앱으로 대화식으로 스윕을 생성합니다.

1. W&B 앱에서 W&B **Project**로 이동합니다.
2. 왼쪽 패널에서 스윕 아이콘(빗자루 이미지)을 선택합니다.
3. 다음으로 **스윕 생성** 버튼을 선택합니다.
4. **Launch 🚀 구성** 버튼을 클릭합니다.
5. **작업** 드롭다운 메뉴에서 스윕을 생성하려는 작업 이름과 작업 버전을 선택합니다.
6. **대기열** 드롭다운 메뉴를 사용하여 스윕을 실행할 대기열을 선택합니다.
7. **작업 우선 순위** 드롭다운을 사용하여 **Launch** 작업의 우선 순위를 지정합니다. **Launch** 작업의 우선 순위는 **Launch** 대기열이 우선 순위 지정을 지원하지 않는 경우 "보통"으로 설정됩니다.
8. (선택 사항) **Run** 또는 스윕 스케줄러에 대한 재정의 인수를 구성합니다. 예를 들어 스케줄러 재정의를 사용하여 `num_workers`를 사용하여 스케줄러가 관리하는 동시 **Run** 수를 구성합니다.
9. (선택 사항) **대상 프로젝트** 드롭다운 메뉴를 사용하여 스윕을 저장할 **Project**를 선택합니다.
10. **저장**을 클릭합니다.
11. **스윕 시작**을 선택합니다.

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI를 사용하여 프로그래밍 방식으로 **Launch**를 통해 W&B **Sweep**를 생성합니다.

1. **Sweep** 구성 생성
2. 스윕 구성 내에서 전체 작업 이름을 지정합니다.
3. 스윕 에이전트를 초기화합니다.

{{% alert %}}
1단계와 3단계는 일반적으로 W&B **Sweep**를 생성할 때 수행하는 단계와 동일합니다.
{{% /alert %}}

예를 들어 다음 코드 조각에서는 작업 값에 대해 `'wandb/jobs/Hello World 2:latest'`를 지정합니다.

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs 를 사용하는 sweep 예제

method: bayes
metric:
  goal: minimize
  name: loss_metric
parameters:
  learning_rate:
    max: 0.02
    min: 0
    distribution: uniform
  epochs:
    max: 20
    min: 0
    distribution: int_uniform

# 선택적 스케줄러 파라미터:

# scheduler:
#   num_workers: 1  # 동시 스윕 runs
#   docker_image: <base image for the scheduler>
#   resource: <ie. local-container...>
#   resource_args:  # runs 에 전달되는 resource 인수
#     env: 
#         - WANDB_API_KEY

# 선택적 Launch 파라미터
# launch: 
#    registry: <registry for image pulling>
```

스윕 구성을 만드는 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ko" >}}) 페이지를 참조하십시오.

4. 다음으로 스윕을 초기화합니다. 구성 파일의 경로, 작업 대기열 이름, W&B **Entity** 및 **Project** 이름을 제공합니다.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B **Sweeps**에 대한 자세한 내용은 [하이퍼파라미터 튜닝]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 챕터를 참조하십시오.

{{% /tab %}}
{{< /tabpane >}}


## 사용자 정의 스윕 스케줄러 생성하기
W&B 스케줄러 또는 사용자 정의 스케줄러를 사용하여 사용자 정의 스윕 스케줄러를 생성합니다.

{{% alert %}}
스케줄러 작업을 사용하려면 wandb cli 버전 >= `0.15.4`가 필요합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B 스윕 스케줄링 로직을 작업으로 사용하여 **Launch** 스윕을 생성합니다.
  
  1. 공용 wandb/sweep-jobs **Project**에서 Wandb 스케줄러 작업을 식별하거나 작업 이름을 사용합니다.
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. 아래 예제와 같이 이 이름을 가리키는 `job` 키가 포함된 추가 `scheduler` 블록이 있는 구성 yaml을 구성합니다.
  3. 새 구성으로 `wandb launch-sweep` 명령을 사용합니다.


구성 예:
```yaml
# launch-sweep-config.yaml  
description: 스케줄러 작업을 사용하는 Launch sweep 구성
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8개의 동시 스윕 runs 허용

# 스윕 runs 가 실행할 트레이닝/튜닝 작업
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  스케줄러 작업을 만들어 사용자 정의 스케줄러를 만들 수 있습니다. 이 가이드에서는 로깅을 더 많이 제공하기 위해 `WandbScheduler`를 수정합니다.

  1. `wandb/launch-jobs` 리포지토리(특히 `wandb/launch-jobs/jobs/sweep_schedulers`)를 복제합니다.
  2. 이제 원하는 로깅 증가를 달성하기 위해 `wandb_scheduler.py`를 수정할 수 있습니다. 예: 함수 `_poll`에 로깅을 추가합니다. 이것은 새 스윕 **Run**을 시작하기 전에 폴링 주기(구성 가능한 타이밍)마다 한 번 호출됩니다.
  3. 수정된 파일을 실행하여 작업을 만듭니다. `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. UI 또는 이전 호출의 출력에서 생성된 작업의 이름을 식별합니다. 이는 코드-아티팩트 작업입니다(달리 지정되지 않은 경우).
  5. 이제 스케줄러가 새 작업을 가리키는 스윕 구성을 만듭니다.

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna는 지정된 모델에 대한 최적의 하이퍼파라미터를 찾기 위해 다양한 알고리즘을 사용하는 하이퍼파라미터 최적화 프레임워크입니다 (W&B와 유사). [샘플링 알고리즘](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) 외에도 Optuna는 성능이 낮은 **Run**을 조기에 종료하는 데 사용할 수 있는 다양한 [Pruning 알고리즘](https://optuna.readthedocs.io/en/stable/reference/pruners.html)을 제공합니다. 이는 많은 수의 **Run**을 실행할 때 특히 유용하며 시간과 리소스를 절약할 수 있습니다. 클래스는 고도로 구성 가능하며 구성 파일의 `scheduler.settings.pruner/sampler.args` 블록에 예상되는 파라미터를 전달하기만 하면 됩니다.

Optuna의 스케줄링 로직을 작업과 함께 사용하여 **Launch** 스윕을 생성합니다.

1. 먼저 사용자 고유의 작업을 만들거나 미리 빌드된 Optuna 스케줄러 이미지 작업을 사용합니다.
    * 사용자 고유의 작업을 만드는 방법에 대한 예는 [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) 리포지토리를 참조하십시오.
    * 미리 빌드된 Optuna 이미지를 사용하려면 `wandb/sweep-jobs` **Project**에서 `job-optuna-sweep-scheduler`로 이동하거나 작업 이름 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`를 사용할 수 있습니다.

2. 작업을 만든 후에는 이제 스윕을 만들 수 있습니다. Optuna 스케줄러 작업을 가리키는 `job` 키가 있는 `scheduler` 블록이 포함된 스윕 구성을 구성합니다(아래 예제).

```yaml
  # optuna_config_basic.yaml
  description: 기본적인 Optuna 스케줄러
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # 이미지에서 가져온 스케줄러 작업에 필요함
    num_workers: 2

    # optuna 특정 설정
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # runs 의 75% 종료
          n_warmup_steps: 10  # 처음 x 단계에서는 가지치기 해제

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

  3. 마지막으로, **Launch-Sweep** 명령으로 활성 대기열에 스윕을 실행합니다.
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna 스윕 스케줄러 작업의 정확한 구현은 [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)를 참조하십시오. Optuna 스케줄러로 가능한 작업에 대한 자세한 예제는 [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)를 확인하십시오.
{{% /tab %}}
{{< /tabpane >}}

 사용자 정의 스윕 스케줄러 작업으로 가능한 작업의 예는 `jobs/sweep_schedulers` 아래의 [wandb/launch-jobs](https://github.com/wandb/launch-jobs) 리포지토리에서 확인할 수 있습니다. 이 가이드에서는 공개적으로 사용 가능한 **Wandb 스케줄러 작업**을 사용하는 방법을 보여주고 사용자 정의 스윕 스케줄러 작업을 만드는 프로세스를 보여줍니다.

## Launch에서 스윕을 재개하는 방법
  이전에 시작된 스윕에서 **Launch-Sweep**를 재개할 수도 있습니다. 하이퍼파라미터와 트레이닝 작업은 변경할 수 없지만 스케줄러 관련 파라미터와 푸시되는 대기열은 변경할 수 있습니다.

{{% alert %}}
초기 스윕에서 '최신'과 같은 에일리어스가 있는 트레이닝 작업을 사용한 경우, 마지막 **Run** 이후 최신 작업 버전이 변경되었다면 재개하면 다른 결과가 발생할 수 있습니다.
{{% /alert %}}

  1. 이전에 실행된 **Launch** 스윕의 스윕 이름/ID를 식별합니다. 스윕 ID는 W&B 앱의 **Project**에서 찾을 수 있는 8자 문자열(예: `hhd16935`)입니다.
  2. 스케줄러 파라미터를 변경하는 경우 업데이트된 구성 파일을 구성합니다.
  3. 터미널에서 다음 명령을 실행합니다. `<` 및 `>`로 묶인 콘텐츠를 정보로 바꿉니다.

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```
