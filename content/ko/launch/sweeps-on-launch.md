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

W&B Launch를 사용하여 하이퍼파라미터 튜닝 작업( [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}))을 생성합니다. Launch에서 Sweeps를 사용하면 스윕 스케줄러가 스윕할 지정된 하이퍼파라미터와 함께 Launch Queue로 푸시됩니다. 스윕 스케줄러는 에이전트가 선택함에 따라 시작되어 선택한 하이퍼파라미터로 스윕 run을 동일한 Queue로 시작합니다. 이는 스윕이 완료되거나 중지될 때까지 계속됩니다.

기본 W&B 스윕 스케줄링 엔진을 사용하거나 자체 사용자 정의 스케줄러를 구현할 수 있습니다.

1. 표준 스윕 스케줄러: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 제어하는 기본 W&B 스윕 스케줄링 엔진을 사용합니다. 친숙한 `bayes`, `grid` 및 `random` 메소드를 사용할 수 있습니다.
2. 사용자 정의 스윕 스케줄러: 스윕 스케줄러가 작업으로 실행되도록 구성합니다. 이 옵션을 사용하면 완벽하게 사용자 정의할 수 있습니다. 표준 스윕 스케줄러를 확장하여 더 많은 로깅을 포함하는 방법의 예는 아래 섹션에서 찾을 수 있습니다.

{{% alert %}}
이 가이드에서는 W&B Launch가 이전에 구성되었다고 가정합니다. W&B Launch가 구성되지 않은 경우 Launch 설명서의 [시작 방법]({{< relref path="./#how-to-get-started" lang="ko" >}}) 섹션을 참조하세요.
{{% /alert %}}

{{% alert %}}
Launch에서 Sweeps를 처음 사용하는 경우 'basic' 메소드를 사용하여 Launch에서 스윕을 생성하는 것이 좋습니다. 표준 W&B 스케줄링 엔진이 요구 사항을 충족하지 못하는 경우 Launch 스케줄러에서 사용자 정의 스윕을 사용합니다.
{{% /alert %}}

## W&B 표준 스케줄러로 스윕 생성
Launch로 W&B Sweeps를 생성합니다. W&B App을 사용하여 대화식으로 또는 W&B CLI를 사용하여 프로그래밍 방식으로 스윕을 생성할 수 있습니다. 스케줄러를 사용자 정의하는 기능을 포함하여 Launch 스윕의 고급 구성은 CLI를 사용하십시오.

{{% alert %}}
W&B Launch로 스윕을 생성하기 전에 먼저 스윕할 작업을 생성해야 합니다. 자세한 내용은 [작업 생성]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ko" >}}) 페이지를 참조하십시오.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App을 사용하여 대화식으로 스윕을 생성합니다.

1. W&B App에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 스윕 아이콘(빗자루 이미지)을 선택합니다.
3. 다음으로 **스윕 생성** 버튼을 선택합니다.
4. **Launch 구성 🚀** 버튼을 클릭합니다.
5. **작업** 드롭다운 메뉴에서 스윕을 생성할 작업 이름과 작업 버전을 선택합니다.
6. **Queue** 드롭다운 메뉴를 사용하여 스윕을 실행할 Queue를 선택합니다.
7. **작업 우선 순위** 드롭다운을 사용하여 Launch 작업의 우선 순위를 지정합니다. Launch Queue가 우선 순위 지정을 지원하지 않으면 Launch 작업의 우선 순위가 "보통"으로 설정됩니다.
8. (선택 사항) Run 또는 스윕 스케줄러에 대한 재정의 인수를 구성합니다. 예를 들어 스케줄러 재정의를 사용하여 스케줄러가 관리하는 동시 Run 수를 `num_workers`를 사용하여 구성합니다.
9. (선택 사항) **대상 프로젝트** 드롭다운 메뉴를 사용하여 스윕을 저장할 프로젝트를 선택합니다.
10. **저장**을 클릭합니다.
11. **스윕 시작**을 선택합니다.

{{< img src="/images/launch/create_sweep_with_launch.png" alt="" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI를 사용하여 프로그래밍 방식으로 Launch로 W&B 스윕을 생성합니다.

1. 스윕 구성을 만듭니다.
2. 스윕 구성 내에서 전체 작업 이름을 지정합니다.
3. 스윕 에이전트를 초기화합니다.

{{% alert %}}
1단계와 3단계는 일반적으로 W&B 스윕을 생성할 때 수행하는 단계와 동일합니다.
{{% /alert %}}

예를 들어 다음 코드 조각에서는 작업 값으로 `'wandb/jobs/Hello World 2:latest'`를 지정합니다.

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs를 사용한 스윕 예제

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
#   num_workers: 1  # 동시 스윕 Runs
#   docker_image: <스케줄러의 기본 이미지>
#   resource: <예: local-container...>
#   resource_args:  # Runs에 전달되는 리소스 인수
#     env: 
#         - WANDB_API_KEY

# 선택적 Launch 파라미터
# launch: 
#    registry: <이미지 풀링 레지스트리>
```

스윕 구성 생성 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ko" >}}) 페이지를 참조하십시오.

4. 다음으로 스윕을 초기화합니다. 구성 파일의 경로, 작업 Queue 이름, W&B 엔티티 및 프로젝트 이름을 제공합니다.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps에 대한 자세한 내용은 [하이퍼파라미터 튜닝]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 챕터를 참조하십시오.

{{% /tab %}}
{{< /tabpane >}}

## 사용자 정의 스윕 스케줄러 생성
W&B 스케줄러 또는 사용자 정의 스케줄러로 사용자 정의 스윕 스케줄러를 생성합니다.

{{% alert %}}
스케줄러 작업을 사용하려면 wandb CLI 버전 >= `0.15.4`가 필요합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B 스윕 스케줄링 로직을 작업으로 사용하여 Launch 스윕을 생성합니다.
  
  1. 공개 wandb/sweep-jobs 프로젝트에서 Wandb 스케줄러 작업을 식별하거나 작업 이름을 사용합니다.
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. 아래 예와 같이 이 이름을 가리키는 `job` 키가 포함된 추가 `scheduler` 블록이 있는 구성 yaml을 구성합니다.
  3. 새 구성으로 `wandb launch-sweep` 명령을 사용합니다.

예제 구성:
```yaml
# launch-sweep-config.yaml  
description: 스케줄러 작업을 사용하여 Launch 스윕 구성
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8개의 동시 스윕 Runs를 허용합니다.

# 스윕 Runs가 실행할 트레이닝/튜닝 작업
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  사용자 정의 스케줄러는 스케줄러 작업을 만들어 생성할 수 있습니다. 이 가이드에서는 로깅을 더 많이 제공하기 위해 `WandbScheduler`를 수정합니다.

  1. `wandb/launch-jobs` 리포지토리를 복제합니다(특히: `wandb/launch-jobs/jobs/sweep_schedulers`).
  2. 이제 `wandb_scheduler.py`를 수정하여 원하는 로깅 증가를 달성할 수 있습니다. 예: 함수 `_poll`에 로깅을 추가합니다. 이는 새 스윕 Runs를 시작하기 전에 폴링 주기(구성 가능한 타이밍)마다 한 번씩 호출됩니다.
  3. 수정된 파일을 실행하여 작업을 만듭니다. `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. UI 또는 이전 호출의 출력에서 생성된 작업의 이름을 식별합니다. 이는 코드 아티팩트 작업입니다(달리 지정하지 않은 경우).
  5. 이제 스케줄러가 새 작업을 가리키는 스윕 구성을 만듭니다.

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna는 주어진 모델에 대한 최상의 하이퍼파라미터를 찾기 위해 다양한 알고리즘을 사용하는 하이퍼파라미터 최적화 프레임워크입니다(W&B와 유사). [샘플링 알고리즘](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) 외에도 Optuna는 성능이 낮은 Runs를 조기에 종료하는 데 사용할 수 있는 다양한 [가지치기 알고리즘](https://optuna.readthedocs.io/en/stable/reference/pruners.html)도 제공합니다. 이는 많은 수의 Runs를 실행할 때 특히 유용하며 시간과 리소스를 절약할 수 있습니다. 클래스는 고도로 구성 가능하며 구성 파일의 `scheduler.settings.pruner/sampler.args` 블록에서 예상되는 파라미터를 전달하기만 하면 됩니다.

Optuna의 스케줄링 로직을 작업과 함께 사용하여 Launch 스윕을 생성합니다.

1. 먼저 자신의 작업을 만들거나 미리 빌드된 Optuna 스케줄러 이미지 작업을 만듭니다.
    * 자신의 작업을 만드는 방법에 대한 예는 [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) 리포지토리를 참조하십시오.
    * 미리 빌드된 Optuna 이미지를 사용하려면 `wandb/sweep-jobs` 프로젝트에서 `job-optuna-sweep-scheduler`로 이동하거나 작업 이름 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`를 사용할 수 있습니다.

2. 작업을 만든 후 스윕을 만들 수 있습니다. Optuna 스케줄러 작업을 가리키는 `job` 키가 있는 `scheduler` 블록이 포함된 스윕 구성을 만듭니다(아래 예제).

```yaml
  # optuna_config_basic.yaml
  description: 기본 Optuna 스케줄러
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # 이미지에서 제공되는 스케줄러 작업에 필요합니다.
    num_workers: 2

    # optuna 특정 설정
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # Runs의 75% 종료
          n_warmup_steps: 10  # 처음 x단계에서는 가지치기가 꺼집니다.

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

  3. 마지막으로 launch-sweep 명령으로 활성 Queue에 스윕을 시작합니다.

  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

  Optuna 스윕 스케줄러 작업의 정확한 구현은 [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)를 참조하십시오. Optuna 스케줄러로 가능한 작업에 대한 자세한 예는 [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)를 확인하십시오.
{{% /tab %}}
{{< /tabpane >}}

 사용자 정의 스윕 스케줄러 작업으로 가능한 작업의 예는 `jobs/sweep_schedulers` 아래의 [wandb/launch-jobs](https://github.com/wandb/launch-jobs) 리포지토리에서 확인할 수 있습니다. 이 가이드에서는 공개적으로 사용 가능한 **Wandb 스케줄러 작업**을 사용하는 방법과 사용자 정의 스윕 스케줄러 작업을 생성하는 프로세스를 보여줍니다.

## Launch에서 스윕을 재개하는 방법
  이전에 시작된 스윕에서 Launch 스윕을 재개할 수도 있습니다. 하이퍼파라미터와 트레이닝 작업은 변경할 수 없지만 스케줄러별 파라미터와 푸시되는 Queue는 변경할 수 있습니다.

{{% alert %}}
초기 스윕에서 'latest'와 같은 에일리어스가 있는 트레이닝 작업을 사용한 경우 마지막 Run 이후 최신 작업 버전이 변경되면 재개 시 다른 결과가 발생할 수 있습니다.
{{% /alert %}}

  1. 이전에 실행한 Launch 스윕의 스윕 이름/ID를 식별합니다. 스윕 ID는 W&B App의 프로젝트에서 찾을 수 있는 8자 문자열입니다(예: `hhd16935`).
  2. 스케줄러 파라미터를 변경하는 경우 업데이트된 구성 파일을 구성합니다.
  3. 터미널에서 다음 명령을 실행합니다. `<`와 `>`로 묶인 내용을 정보로 바꿉니다.

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```
