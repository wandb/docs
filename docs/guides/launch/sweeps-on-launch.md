---
title: Create sweeps with W&B Launch
description: Launch에서 하이퍼파라미터 Sweeps를 자동화하는 방법을 알아보세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7"/>

W&B Launch로 하이퍼파라미터 튜닝 작업([sweeps](../sweeps/intro.md))을 만드세요. Launch에서 sweeps를 사용하면, 스윕 스케줄러가 지정된 하이퍼파라미터와 함께 Launch Queue로 푸시됩니다. 스윕 스케줄러는 에이전트가 선택함에 따라 시작되며, 선택된 하이퍼파라미터로 스윕 실행을 동일한 큐에 시작합니다. 이것은 스윕이 완료되거나 중지될 때까지 계속됩니다.

기본 W&B Sweep 스케줄링 엔진을 사용하거나 사용자 정의 스케줄러를 구현할 수 있습니다:

1. 표준 스윕 스케줄러: [W&B Sweeps](../sweeps/intro.md)를 제어하는 기본 W&B Sweep 스케줄링 엔진을 사용합니다. 익숙한 `bayes`, `grid`, `random` 메소드가 제공됩니다.
2. 사용자 정의 스윕 스케줄러: 스윕 스케줄러를 작업으로 실행하도록 구성합니다. 이 옵션은 완전한 사용자 정의를 가능하게 합니다. 보다 많은 로그를 포함하도록 표준 스윕 스케줄러를 확장하는 방법은 아래 섹션에서 찾을 수 있습니다.

:::note
이 가이드는 W&B Launch가 이전에 구성되었다고 가정합니다. W&B Launch가 구성되지 않은 경우 Launch 문서의 [시작 방법](./intro.md#how-to-get-started) 섹션을 참조하세요.
:::

:::tip
Launch에서 처음으로 sweeps를 사용하는 사용자라면 'basic' 메소드를 사용하여 스윕을 생성하는 것을 권장합니다. 표준 W&B 스케줄링 엔진이 필요를 충족하지 못할 경우 사용자 정의 sweeps on launch 스케줄러를 사용하세요.
:::

## W&B 표준 스케줄러로 스윕 생성하기
Launch로 W&B Sweeps를 만드세요. W&B App으로 상호작용적으로 스윕을 생성하거나 W&B CLI로 프로그래밍적으로 생성할 수 있습니다. 스케줄러를 사용자 정의할 수 있는 Launch sweeps의 고급 구성의 경우 CLI를 사용하세요.

:::info
W&B Launch로 스윕을 생성하기 전에 먼저 스윕을 수행할 작업을 생성했는지 확인하세요. 자세한 내용은 [Create a Job](./create-launch-job.md) 페이지를 참조하세요.
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B App에서 상호작용적으로 스윕을 생성하세요.

1. W&B App에서 W&B 프로젝트로 이동합니다.  
2. 왼쪽 패널의 스윕 아이콘(빗자루 이미지)을 선택합니다.
3. 다음으로 **Create Sweep** 버튼을 선택합니다.
4. **Configure Launch 🚀** 버튼을 클릭합니다.
5. **Job** 드롭다운 메뉴에서 스윕을 생성할 작업의 이름과 작업 버전을 선택합니다.
6. **Queue** 드롭다운 메뉴를 사용하여 스윕을 실행할 큐를 선택합니다.
8. **Job Priority** 드롭다운을 사용하여 Launch 작업의 우선 순위를 지정합니다. Launch 큐가 우선 순위를 지원하지 않으면 Launch 작업의 우선 순위는 "중간"으로 설정됩니다.
8. (선택 사항) 실행이나 스윕 스케줄러에 대한 인수를 재정의합니다. 예를 들어, 스케줄러 재정의를 사용하여 스케줄러가 관리하는 동시 실행 수를 `num_workers`를 사용하여 구성합니다.
9. (선택 사항) **Destination Project** 드롭다운 메뉴를 사용하여 스윕을 저장할 프로젝트를 선택합니다.
10. **저장**을 클릭합니다.
11. **Launch Sweep**을 선택합니다.

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">

W&B CLI로 Launch로 W&B Sweep을 프로그래밍적으로 생성하세요.

1. 스윕 구성을 생성합니다.
2. 스윕 구성 내에 전체 작업 이름을 지정합니다.
3. 스윕 에이전트를 초기화합니다.

:::info
단계 1과 3은 일반적으로 W&B Sweep을 생성할 때 걸음과 같습니다.
:::

예를 들어, 다음 코드 조각에서는 작업 값으로 `'wandb/jobs/Hello World 2:latest'`를 지정합니다:

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
#   num_workers: 1  # 동시 스윕 실행 수
#   docker_image: <화상 기초 이미지>
#   resource: <ie. local-container...>
#   resource_args:  # run에 전달되는 자원 인수
#     env: 
#         - WANDB_API_KEY

# 선택적 Launch Params
# launch: 
#    registry: <이미지 다운로드 레지스트리>
```

스윕 구성을 만드는 방법에 대한 정보는 [Define sweep configuration](../sweeps/define-sweep-configuration.md) 페이지를 참조하세요.

4. 다음으로, 스윕을 초기화합니다. 설정 파일의 경로, 작업 큐의 이름, W&B 엔티티, 그리고 프로젝트 이름을 제공합니다.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B 게기 정보 Sweeps에 대한 자세한 내용은 [Tune Hyperparameters](../sweeps/intro.md) 챕터를 참조하세요.

</TabItem>

</Tabs>

## 사용자 정의 스윕 스케줄러 생성하기
W&B 스케줄러 또는 사용자 정의 스케줄러로 사용자 정의 스윕 스케줄러를 생성합니다.

:::info
스케줄러 작업을 사용하는 데는 wandb cli 버전이 `0.15.4` 이상이어야 합니다.
:::

<Tabs
  defaultValue="wandb-scheduler"
  values={[
    {label: 'Wandb scheduler', value: 'wandb-scheduler'},
    {label: 'Optuna scheduler', value: 'optuna-scheduler'},
    {label: 'Custom scheduler', value: 'custom-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

  W&B 스윕 스케줄링 로직을 작업으로 해서 Launch 스윕을 생성하세요.
  
  1. 공용 wandb/sweep-jobs 프로젝트에서 Wandb 스케줄러 작업을 식별하거나 작업 이름을 사용하세요:
    `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. yaml 구성을 추가 `scheduler` 블록과 함께 구성하여 이 이름을 가리키는 `job` 키를 포함합니다. 아래 예를 참조하세요.
  3. 새 구성으로 `wandb launch-sweep` 코맨드를 사용하세요.


구성 예:
```yaml
# launch-sweep-config.yaml  
description: 스케줄러 작업을 사용한 Launch 스윕 구성
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8 개의 동시 스윕 실행 허용

# 스윕 실행을 수행할 트레이닝/튜닝 작업
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```

  </TabItem>
  <TabItem value="custom-scheduler">

  사용자 정의 스케줄러는 스케줄러 작업을 생성하여 만들 수 있습니다. 이 가이드의 목적으로 `WandbScheduler`를 수정하여 더 많은 로그를 제공하도록 하겠습니다.

  1. `wandb/launch-jobs` 리포를 복제합니다 (특히: `wandb/launch-jobs/jobs/sweep_schedulers`).
  2. 이제 원하는 로그 증가를 달성하기 위해 `wandb_scheduler.py`를 수정할 수 있습니다. 예: 기능 `_poll`에 로그를 추가합니다. 이는 새로운 스윕 실행을 시작하기 전에 매 폴링 주기(구성 가능한 타이밍)마다 한 번 호출됩니다.
  3. 다음 명령어로 작업을 생성하기 위해 수정된 파일을 실행하세요: `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 생성된 작업의 이름을 확인합니다. 이는 UI 또는 이전 호출의 출력에서 확인할 수 있으며, 별도로 지정하지 않는 한 코드 아티팩트 작업이 될 것입니다.
  5. 이제 스케줄러가 새로운 작업을 가리키는 스윕 구성을 생성하세요!

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

  Optuna는 주어진 모델에 대해 최고의 하이퍼파라미터를 찾기 위해 다양한 알고리즘을 사용하는 하이퍼파라미터 최적화 프레임워크입니다(비슷하게 W&B). [샘플링 알고리즘](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) 외에도 Optuna는 실행이 좋지 않을 때 일찍 종료하는 데 사용할 수 있는 다양한 [프루닝 알고리즘](https://optuna.readthedocs.io/en/stable/reference/pruners.html)을 제공합니다. 이는 많은 실행을 할 때 특히 유용하며 시간과 자원을 절약할 수 있습니다. 클래스는 매우 구성 가능하며, 설정 파일의 `scheduler.settings.pruner/sampler.args` 블록에 예상 파라미터를 전달하기만 하면 됩니다.

Optuna의 스케줄링 로직을 사용한 Launch 스윕을 작업으로 생성하세요.

1. 먼저, 자신의 작업을 생성하거나 사전 구축된 Optuna 스케줄러 이미지 작업을 사용하세요. 
    * 자신의 작업을 생성하는 방법에 대한 예는 [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) 리포를 참고하세요.
    * 사전 구축된 Optuna 이미지를 사용하려면 `wandb/sweep-jobs` 프로젝트의 `job-optuna-sweep-scheduler`로 이동하거나 다음 작업 이름을 사용하세요: `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`. 
    

2. 작업을 생성한 후, 이제 스윕을 만들 수 있습니다. Optuna 스케줄러 작업을 가리키는 `job` 키가 포함된 `scheduler` 블록을 포함하는 스윕 구성을 생성합니다(아래 예):

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
    resource: local-container  # 이미지에서 소스된 스케줄러 작업에 필요
    num_workers: 2

    # optuna 특정 설정
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # 75%의 실행 종료
          n_warmup_steps: 10  # 초기 x 단계 동안 프루닝 비활성화

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```

3. 마지막으로, launch-sweep 코맨드를 사용하여 활성 큐에 스윕을 시작하세요:
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```

Optuna 스윕 스케줄러 작업의 정확한 구현은 [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)를 참조하세요. Optuna 스케줄러를 사용하여 가능한 작업의 더 많은 예시는 [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)를 확인하세요.

  </TabItem>
</Tabs>

사용자 정의 스윕 스케줄러 작업으로 가능한 예시는 `jobs/sweep_schedulers` 아래에 있는 [wandb/launch-jobs](https://github.com/wandb/launch-jobs) 리포에서 사용할 수 있습니다. 이 가이드에서는 공개적으로 이용 가능한 **Wandb Scheduler Job**을 사용하는 방법을 보여주며, 사용자 정의 스윕 스케줄러 작업을 생성하는 프로세스를 소개합니다.

## Launch에서 스윕 다시 시작하는 방법
Launch에서 이전에 실행된 스윕을 통해 launch-sweep을 다시 시작할 수도 있습니다. 하이퍼파라미터와 트레이닝 작업은 변경할 수 없지만, 스캐줄러 전용 파라미터와 푸시할 큐는 변경할 수 있습니다.

:::info
초기 스윕에서 'latest'와 같은 에일리어스를 사용한 트레이닝 작업을 사용할 경우, 최신 작업 버전이 마지막 실행 이후로 변경된 경우 다른 결과를 낼 수 있습니다.
:::

1. 이전에 실행된 launch sweep의 스윕 이름/ID를 식별하세요. 스윕 ID는 W&B App의 프로젝트에서 찾을 수 있는 여덟 자 문자열입니다 (예를 들어, `hhd16935`).
2. 스케줄러 파라미터를 변경하려면, 업데이트된 구성 파일을 생성하세요.
3. 터미널에서 다음 명령어를 실행합니다. `<` 및 `>` 사이에 감싸진 내용을 해당 정보로 대체하세요:

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```