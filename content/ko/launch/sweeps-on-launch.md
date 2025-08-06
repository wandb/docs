---
title: W&B Launch로 스윕 생성하기
description: Launch 에서 하이퍼파라미터 스윕을 자동화하는 방법을 알아보세요.
menu:
  launch:
    identifier: ko-launch-sweeps-on-launch
    parent: launch
url: guides/launch/sweeps-on-launch
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1WxLKaJlltThgZyhc7dcZhDQ6cjVQDfil#scrollTo=AFEzIxA6foC7" >}}

W&B Launch에서 [sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 사용해 하이퍼파라미터 튜닝 작업을 생성합니다. Launch에서 sweeps를 사용하면 지정한 하이퍼파라미터에 따라 sweep 스케줄러가 Launch Queue에 등록됩니다. 이 스케줄러는 에이전트가 잡아서 실행을 시작하며, 선택된 하이퍼파라미터로 sweep run을 같은 큐에 투입합니다. 이 과정은 sweep이 완료되거나 중지될 때까지 계속됩니다.

W&B Sweep의 기본 스케줄링 엔진을 사용할 수도 있고, 직접 커스텀 스케줄러를 구현할 수도 있습니다:

1. 표준 스윕 스케줄러: [W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})를 제어하는 기본 W&B Sweep 스케줄링 엔진을 사용하세요. 익숙한 `bayes`, `grid`, `random` 메소드가 지원됩니다.
2. 커스텀 스윕 스케줄러: 직접 만든 스케줄러를 job으로 등록해 사용하세요. 이 옵션은 완전한 커스터마이징이 가능합니다. 스케줄러에 추가 로그를 남기는 예시는 아래에서 확인하실 수 있습니다.

{{% alert %}}
이 가이드는 W&B Launch가 미리 설정되어 있다고 가정합니다. W&B Launch 설정이 되어 있지 않으시다면, launch 문서의 [시작 방법]({{< relref path="./#how-to-get-started" lang="ko" >}}) 섹션을 참고하세요.
{{% /alert %}}

{{% alert %}}
처음으로 launch에서 sweeps를 사용하신다면 'basic' 메소드로 sweep을 생성하는 것을 권장합니다. 표준 W&B 스케줄링 엔진이 요구 사항을 충족하지 못하는 경우에만 커스텀 sweeps on launch 스케줄러를 사용하세요.
{{% /alert %}}

## W&B 표준 스케줄러로 sweep 생성하기
Launch에서 W&B Sweeps를 생성하세요. W&B App에서 상호작용적으로 sweep을 만들 수도 있고, W&B CLI를 통해 프로그래밍 방식으로도 생성할 수 있습니다. Launch sweep의 고급 설정(스케줄러 커스터마이징 등)은 CLI에서 가능합니다.

{{% alert %}}
W&B Launch로 sweep을 만들기 전에, sweep 대상 job을 먼저 생성해야 합니다. 자세한 내용은 [Create a Job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ko" >}}) 페이지를 참고하세요.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B app" %}}

W&B App에서 상호작용적으로 sweep을 생성하세요.

1. W&B App에서 본인의 W&B 프로젝트로 이동합니다.  
2. 왼쪽 패널에서 sweeps 아이콘(빗자루 모양)을 선택합니다.
3. **Create Sweep** 버튼을 클릭합니다.
4. **Configure Launch** 버튼을 클릭하세요.
5. **Job** 드롭다운에서 sweep을 생성할 job과 해당 버전을 선택하세요.
6. **Queue** 드롭다운을 통해 sweep을 실행할 큐를 선택하세요.
7. **Job Priority**(작업 우선순위) 드롭다운에서 launch job의 우선순위를 정하세요. launch queue가 우선순위 기능을 지원하지 않을 경우, 기본값은 "Medium"입니다.
8. (선택) run 또는 sweep 스케줄러를 위한 override arg를 추가 설정할 수 있습니다. 예를 들어, scheduler override에서 스케줄러가 관리할 동시 실행 run 수를 `num_workers`로 지정할 수 있습니다.
9. (선택) **Destination Project** 드롭다운에서 sweep 결과를 저장할 프로젝트를 선택하세요.
10. **Save**를 클릭합니다.
11. **Launch Sweep**을 선택하세요.

{{< img src="/images/launch/create_sweep_with_launch.png" alt="Launch sweep configuration" >}}

{{% /tab %}}
{{% tab "CLI" %}}

W&B CLI를 통해 프로그래밍적으로 Launch에서 W&B Sweep을 만드세요.

1. Sweep 구성을 생성합니다.
2. sweep 구성 내에 사용할 job의 전체 이름을 지정합니다.
3. sweep 에이전트를 초기화하세요.

{{% alert %}}
1번과 3번 단계는 W&B Sweep을 만드는 일반 과정과 동일합니다.
{{% /alert %}}

아래 코드 예시에서는 job 값으로 `'wandb/jobs/Hello World 2:latest'`를 지정한 예입니다:

```yaml
# launch-sweep-config.yaml

job: 'wandb/jobs/Hello World 2:latest'
description: sweep examples using launch jobs

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

# Optional scheduler parameters:

# scheduler:
#   num_workers: 1  # 동시에 실행할 sweep run 수
#   docker_image: <스케줄러의 베이스 이미지>
#   resource: <예: local-container...>
#   resource_args:  # run에 전달되는 리소스 인수
#     env: 
#         - WANDB_API_KEY

# Optional Launch Params
# launch: 
#    registry: <이미지 pull을 위한 레지스트리>
```

sweep 구성 작성 방법은 [Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration.md" lang="ko" >}}) 페이지를 참고하세요.

4. 다음으로 sweep을 초기화합니다. 구성 파일 경로, job queue 이름, W&B entity, project 이름을 입력하세요.

```bash
wandb launch-sweep <path/to/yaml/file> --queue <queue_name> --entity <your_entity>  --project <project_name>
```

W&B Sweeps에 대한 자세한 내용은 [Tune Hyperparameters]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 챕터를 참고하세요.

{{% /tab %}}
{{< /tabpane >}}


## 커스텀 sweep 스케줄러 만들기
W&B 스케줄러나 직접 만든 커스텀 스케줄러를 사용해 커스텀 sweep 스케줄러를 생성하세요.

{{% alert %}}
스케줄러 job 사용은 wandb CLI 버전이 `0.15.4` 이상이어야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab "W&B scheduler" %}}
  W&B sweep 스케줄링 로직을 job으로 활용해 launch sweep을 만드세요.
  
  1. 공개된 wandb/sweep-jobs 프로젝트에서 Wandb scheduler job을 찾거나, job 이름 `'wandb/sweep-jobs/job-wandb-sweep-scheduler:latest'`를 사용하세요.
  2. `scheduler` 블록에 이 job 이름을 가리키는 `job` 키를 포함한 추가 설정 yaml을 만듭니다. 예시는 아래와 같습니다.
  3. 새로운 config와 함께 `wandb launch-sweep` 명령어를 사용하세요.


예시 config:
```yaml
# launch-sweep-config.yaml  
description: Launch sweep config using a scheduler job
scheduler:
  job: wandb/sweep-jobs/job-wandb-sweep-scheduler:latest
  num_workers: 8  # 동시에 실행할 sweep run 수

# sweep에서 실행할 트레이닝/튜닝 job
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
method: grid
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
```
{{% /tab %}}
{{% tab "Custom scheduler" %}}
  스케줄러-job을 만들어 커스텀 스케줄러를 생성할 수 있습니다. 이 가이드에서는 `WandbScheduler`를 수정해 로그를 더 남기는 방법을 예로 들겠습니다.

  1. `wandb/launch-jobs` 저장소(특히 `wandb/launch-jobs/jobs/sweep_schedulers`)를 클론하세요.
  2. `wandb_scheduler.py` 파일을 수정해 로그 수준을 높입니다. 예시는 `_poll` 함수에 로그 추가입니다. 이 함수는 새로운 sweep run을 실행하기 전, polling 주기마다 한 번씩 호출됩니다(주기 조절 가능).
  3. 수정한 파일을 사용해 job을 생성하세요. 예시: `python wandb_scheduler.py --project <project> --entity <entity> --name CustomWandbScheduler`
  4. 생성된 job의 이름을 UI 또는 실행 결과에서 확인합니다. 별도 지정이 없는 한 code-artifact job으로 등록됩니다.
  5. 스케줄러가 새 job을 가리키도록 sweep 구성을 작성하세요.

```yaml
...
scheduler:
  job: '<entity>/<project>/job-CustomWandbScheduler:latest'
...
```

{{% /tab %}}
{{% tab "Optuna scheduler" %}}

  Optuna는 다양한 알고리즘으로 최적의 하이퍼파라미터를 찾을 수 있도록 하는 하이퍼파라미터 최적화 프레임워크입니다(W&B와 유사). [샘플링 알고리즘](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) 외에도, Optuna는 부진한 run을 조기에 중단할 수 있는 [pruning 알고리즘](https://optuna.readthedocs.io/en/stable/reference/pruners.html)도 제공합니다. run의 개수가 많을 때 시간과 리소스를 아낄 수 있습니다. 필요한 파라미터를 `scheduler.settings.pruner/sampler.args` 블록에 전달해 자유롭게 조정 가능합니다.



Optuna 스케줄러 로직을 job으로 사용해 launch sweep을 생성하세요.

1. 직접 job을 만들거나, 미리 만들어진 Optuna 스케줄러 이미지 job을 사용하세요.
    * 직접 job을 만드는 방법은 [`wandb/launch-jobs`](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers) repo 예시를 참고하세요.
    * 미리 만들어진 Optuna 이미지를 쓰려면 `wandb/sweep-jobs` 프로젝트의 `job-optuna-sweep-scheduler`로 이동하거나 다음 job 이름을 사용하세요: `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest`.
    

2. job을 생성한 뒤에는 이제 sweep을 만들 수 있습니다. 예시처럼 `scheduler` 블록에 Optuna 스케줄러 job을 가리키는 `job` 키를 포함한 sweep config를 작성하세요.

```yaml
  # optuna_config_basic.yaml
  description: A basic Optuna scheduler
  job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
  run_cap: 5
  metric:
    name: epoch/val_loss
    goal: minimize

  scheduler:
    job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
    resource: local-container  # 이미지 기반 스케줄러 job에 필요
    num_workers: 2

    # optuna 전용 설정
    settings:
      pruner:
        type: PercentilePruner
        args:
          percentile: 25.0  # run의 75%를 중단
          n_warmup_steps: 10  # 첫 x step에서는 pruning 비활성화

  parameters:
    learning_rate:
      min: 0.0001
      max: 0.1
  ```


  3. 마지막으로 다음 명령어로 실행 중인 queue에 sweep을 런칭하세요:
  
  ```bash
  wandb launch-sweep <config.yaml> -q <queue> -p <project> -e <entity>
  ```


  Optuna sweep 스케줄러 job의 실제 구현 코드는 [wandb/launch-jobs](https://github.com/wandb/launch-jobs/blob/main/jobs/sweep_schedulers/optuna_scheduler/optuna_scheduler.py)에서 확인할 수 있으며, Optuna 스케줄러로 할 수 있는 다양한 예시는 [wandb/examples](https://github.com/wandb/examples/tree/master/examples/launch/launch-sweeps/optuna-scheduler)에서도 참고할 수 있습니다.
{{% /tab %}}
{{< /tabpane >}}

커스텀 sweep 스케줄러 job의 다양한 예시는 [wandb/launch-jobs](https://github.com/wandb/launch-jobs) repo의 `jobs/sweep_schedulers`에서 확인할 수 있습니다. 이 가이드에서는 공개된 **Wandb Scheduler Job** 사용법을 비롯해, 커스텀 sweep 스케줄러 job을 만드는 기본적인 프로세스를 소개합니다.


## Launch에서 sweep 재시작(Resume) 방법
이전에 실행한 launch-sweep에서 이어서 재시작할 수도 있습니다. 하이퍼파라미터나 트레이닝 job은 변경할 수 없지만, 스케줄러 전용 파라미터와 sweep이 투입될 queue는 수정할 수 있습니다.

{{% alert %}}
초기 sweep에서 'latest' 등 에일리어스가 포함된 트레이닝 job을 사용한 경우, 마지막 run 이후 job의 최신 버전이 변경되었다면 결과가 달라질 수 있습니다.
{{% /alert %}}

1. 이전에 실행한 launch sweep의 sweep 이름 또는 ID를 찾으세요. sweep ID는 8자리 문자열이며(예: `hhd16935`), W&B App에 있는 프로젝트에서 확인할 수 있습니다.
2. 스케줄러 파라미터를 바꾸고 싶다면, 새로운 config 파일을 작성하세요.
3. 터미널에서 아래 명령어를 실행합니다. `<`와 `>`로 감싸진 부분은 본인의 정보로 바꿔주세요:

```bash
wandb launch-sweep <optional config.yaml> --resume_id <sweep id> --queue <queue_name>
```