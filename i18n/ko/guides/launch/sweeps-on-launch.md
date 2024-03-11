---
description: Discover how to automate hyperparamter sweeps on launch.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Sweeps on Launch

<CTAButtons colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7"/>

W&B Launch를 사용하여 하이퍼파라미터 튜닝 작업([스윕](../sweeps/intro.md))을 생성하세요. 스윕을 실행하면 지정된 하이퍼파라미터로 스윕을 수행하는 스윕 스케줄러가 Launch Queue에 푸시됩니다. 에이전트가 스케줄러를 선택하면 스윕 스케줄러가 시작되며, 선택된 하이퍼파라미터로 동일한 큐에 스윕 실행이 발생합니다. 이 과정은 스윕이 완료되거나 중단될 때까지 계속됩니다.

기본 W&B 스윕 스케줄링 엔진을 사용하거나 사용자 정의 스케줄러를 구현할 수 있습니다:

1. 표준 스윕 스케줄러: [W&B 스윕](../sweeps/intro.md)을 제어하는 기본 W&B 스윕 스케줄링 엔진을 사용합니다. 익숙한 `bayes`, `grid`, `random` 메소드를 사용할 수 있습니다.
2. 사용자 정의 스윕 스케줄러: 스케줄러를 작업으로 실행하도록 구성합니다. 이 옵션은 전체 사용자 정의를 가능하게 합니다. 아래 섹션에서 로깅을 더 추가하여 표준 스윕 스케줄러를 확장하는 방법의 예를 확인할 수 있습니다.
 
:::note
이 가이드는 W&B Launch가 이전에 설정되었다고 가정합니다. W&B Launch가 설정되지 않은 경우, launch 문서의 [시작 방법](./intro.md#how-to-get-started) 섹션을 참조하십시오.
:::

:::tip
스윕을 실행하는 데 처음인 사용자라면 'basic' 방법을 사용하여 스윕을 생성하는 것이 좋습니다. 표준 W&B 스케줄링 엔진이 요구 사항을 충족시키지 못할 때 사용자 정의 스윕 스케줄러를 사용하세요.
:::

## W&B 표준 스케줄러로 스윕 생성하기
Launch를 통해 W&B 스윕을 생성하세요. W&B App이나 W&B CLI를 사용하여 상호작용적으로 또는 프로그래밍 방식으로 스윕을 생성할 수 있습니다. 스케줄러를 사용자 정의하는 등 Launch 스윕의 고급 구성을 위해서는 CLI를 사용하세요.

:::info
W&B Launch로 스윕을 생성하기 전에, 먼저 스윕할 작업을 생성해야 합니다. 자세한 내용은 [작업 생성](./create-launch-job.md) 페이지를 참조하세요.
:::


<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B App을 사용하여 상호작용적으로 스윕을 생성합니다.

1. W&B App에서 W&B 프로젝트로 이동하세요.
2. 왼쪽 패널에서 스윕 아이콘(빗자루 이미지)을 선택하세요.
3. **스윕 생성** 버튼을 선택하세요.
4. **Launch 구성 🚀** 버튼을 클릭하세요.
5. **작업** 드롭다운 메뉴에서 스윕을 생성할 작업의 이름과 버전을 선택하세요.
6. **큐** 드롭다운 메뉴를 사용하여 스윕을 실행할 큐를 선택하세요.
8. **작업 우선순위** 드롭다운을 사용하여 런치 작업의 우선순위를 지정하세요. 큐가 우선 순위를 지원하지 않는 경우 런치 작업의 우선순위는 "중간"으로 설정됩니다.
8. (선택 사항) 실행 또는 스윕 스케줄러에 대한 재정의 인수를 구성하세요. 예를 들어, 스케줄러 재정의를 사용하여 스케줄러가 관리하는 동시 실행 수를 `num_workers`를 사용하여 구성합니다.
9. (선택 사항) **대상 프로젝트** 드롭다운 메뉴를 사용하여 스윕을 저장할 프로젝트를 선택하세요.
10. **저장**을 클릭하세요.
11. **스윕 실행**을 선택하세요.

![](/images/launch/create_sweep_with_launch.png)

  </TabItem>
  <TabItem value="cli">

W&B CLI를 사용하여 프로그래밍 방식으로 W&B 스윕을 Launch와 함께 생성하세요.

1. 스윕 구성 생성
2. 스윕 구성 내에서 작업의 전체 이름을 지정하세요
3. 스윕 에이전트 초기화

:::info
1단계와 3단계는 일반적으로 W&B 스윕을 생성할 때 수행하는 단계와 동일합니다.
:::

예를 들어, 다음 코드 조각에서 `'wandb/jobs/Hello World 2:latest'`를 작업 값으로 지정합니다:

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: launch jobs를 사용한 스윕 예시

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
#   num_workers: 1  # 동시 스윕 실행
#   docker_image: <스케줄러의 베이스 이미지>
#   resource: <예: local-container...>
#   resource_args:  # 실행에 전달된 리소스 인수
#     env: 
#         - WANDB_API_KEY

# 선택적 Launch 파라미터
# launch: 
#    registry: <이미지 풀링을 위한 레지스트리>
```

스윕 구성을 생성하는 방법에 대한 자세한 내용은 [스윕 구성 정의](../sweeps/define-sweep-configuration.md) 페이지를 참조하세요.

4. 다음으로, 스윕을 초기화합니다. config 파일 경로, 작업 큐 이름, W&B 엔티티 및 프로젝트 이름을 제공하세요.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B 스윕에 대한 자세한 정보는 [하이퍼파라미터 튜닝하기](../sweeps/intro.md) 챕터를 참조하세요.


</TabItem>

</Tabs>

## 사용자 정의 스윕 스케줄러 생성하기
W&B 스케줄러 또는 사용자 정의 스케줄러를 사용하여 사용자 정의 스윕 스케줄러를 생성하세요.

:::info
스케줄러 작업을 사용하려면 wandb cli 버전이 `0.15.4` 이상이어야 합니다.
:::

<Tabs
  defaultValue="wandb-scheduler"
  values={[
    {label: 'Wandb 스케줄러', value: 'wandb-scheduler'},
    {label: 'Optuna 스케줄러', value: 'optuna-scheduler'},
    {label: '사용자 정의 스케줄러', value: 'custom-scheduler'},
  ]}>
    <TabItem value="wandb-scheduler">

  W&B 스윕 스케줄링 로직을 작업으로 사용하여 런치 스윕을 생성하세요.
  
  1. 공개 wandb/sweep-jobs 프로젝트에서 Wandb 스케줄러 작업을 확인하거나, 작업 이름을 사용하세요:
  `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`
  2. `scheduler` 블록을 추가하여 이 이름을 가리키는 `job` 키가 포함된 구성 yaml을 구성하세요. 아래 예시를 참조하세요.
  3. 새 구성으로 `wandb launch-sweep` 명령어를 사용하세요.


예시 구성:
```yaml
# launch-sweep-config.yaml  
description: 스케줄러 작업을 사용한 Launch 스윕 구성
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 8개의 동시 스윕 실행 허용

# 스윕 실행이 수행할 트레이닝/튜닝 작업
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```

  </TabItem>
  <TabItem value="custom-scheduler">

  스케줄러-작업을 생성하여 사용자 정의 스케줄러를 만들 수 있습니다. 이 가이드에서는 로깅을 더 많이 제공하는 `WandbScheduler`를 수정하는 과정을 보여줍니다.

  1. `wandb/launch-jobs` 리포지토리를 클론하세요(특히: `wandb/launch-jobs/jobs/sweep_schedulers`)
  2. 이제 `wandb_scheduler.py`를 수정하여 원하는 증가된 로깅을 달성할 수 있습니다. 예: `_poll` 함수에 로깅을 추가합니다. 이 함수는 폴링 주기(구성 가능한 타이밍)마다 한 번 호출되며, 새로운 스윕 실행을 시작하기 전에 호출됩니다.
  3. `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`를 사용하여 수정된 파일을 실행하여 작업을 생성하세요.
  4. UI에서 또는 이전 호출의 출력에서 작업 이름을 확인하세요. 이 작업은 코드 아티팩트 작업이 될 것입니다(달리 명시되지 않는 한).
  5. 이제 스케줄러가 새 작업을 가리키는 스윕 구성을 생성하세요!

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

  </TabItem>
  <TabItem value="optuna-scheduler">

  Optuna는 주어진 모델에 대한 최적의 하이퍼파라미터를 찾기 위해 다양한 알고리즘을 사용하는 하이퍼파라미터 최적화 프레임워크입니다(W&B와 유사). [샘플링 알고리즘](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) 외에도 Optuna는 성능이 낮은 실행을 조기에 종료할 수 있는 다양한 [프루닝 알고리즘](https://optuna.readthedocs.io/en/stable/reference/pruners.html)을 제공합니다. 이는 많은 수의 실행을 수행할 때 시간과 리소스를 절약하는 데 특히 유용합니다. 클래스는 매우 구성 가능하며, 구성 파일의 `scheduler.settings.pruner/sampler.args` 블록에 예상되는 파라미터를 전달하기만 하면 됩니다.



Optuna의 스케줄링 로직을 사용하여 작업으로 런치 스윕을 생성하세요.

1. 먼저, 자신만의 작업을 생성하거나 사전 빌드된 Optuna 스케줄러 이미지 작업을 사용하세요.
    * 자신만의 작업을 생성하는 방법에 대한 예제는 [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) 리포지토리를 참조하세요.
    * 사전 빌드된 Optuna 이미지를 사용하려면 `wandb/sweep-jobs` 프로젝트의 `job-optuna-sweep-scheduler`로 이동하거나 `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` 작업 이름을 사용할 수 있습니다.
    

2. 작업을 생성한 후, `scheduler` 블록이 포함된 스윕 구성을 생성하여 Optuna 스케줄러 작업을 가리키는 `job` 키를 포함시킬 수 있습니다(아래 예시 참조).

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
    resource: local-container  # 이미지에서 가져온 스케줄러 작업에 필요
    num_workers: 2

    # optuna 특정 설정
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # 75%의 실행을 종료
          n_warmup_steps: 10  # 처음 x 단계에 대해 프루닝 비활성화

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. 마지막으로, 런치 스윕 명령어를 사용하여 활성 큐에 스윕을 시작하세요:
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```


  Optuna 스윕 스케줄러 작업의 정확한 구현에 대해서는 [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)를 참조하세요. Optuna 스케줄러로 가능한 것의 더 많은 예제는 [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)에서 확인할 수 있습니다.


  </TabItem>
</Tabs>

 사용자 정의 스윕 스케줄러 작업으로 가능한 것의 예제는 [wandb/launch-jobs](https://github.com/wandb/launch-jobs) 리포지토리에서 `jobs/sweep_schedulers` 아래에서 확인할 수 있습니다. 이 가이드는 공개적으로 사용 가능한 **Wandb 스케줄러 작업**을 사용하는 방법을 보여주며, 사용자 정의 스윕 스케줄러 작업을 생성하는 프로세스를 설명합니다.


 ## Launch에서 스윕 재개하는 방법
  이전에 실행된 스윕에서 런치 스윕을 재개하는 것도 가능