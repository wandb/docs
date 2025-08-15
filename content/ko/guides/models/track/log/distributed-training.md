---
title: 분산 트레이닝 실험 로그 남기기
description: 여러 개의 GPU 를 사용한 분산 트레이닝 실험을 W&B 로 로그하세요.
menu:
  default:
    identifier: ko-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

분산 트레이닝 실험에서는 여러 대의 머신 또는 클라이언트를 병렬로 이용해 모델을 트레이닝합니다. W&B를 활용하면 분산 트레이닝 실험을 효과적으로 추적할 수 있습니다. 유스 케이스에 따라 아래 방법 중 하나로 분산 트레이닝 실험을 추적해보세요:

* **단일 프로세스 추적**: W&B로 rank 0 프로세스(“리더” 또는 “코디네이터”라고도 부름)만 추적합니다. 이는 [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) 클래스를 사용할 때 일반적으로 쓰이는 방식입니다.
* **다중 프로세스 추적**: 여러 프로세스를 사용하는 경우 아래 중 한 가지 방식을 선택할 수 있습니다.
   * 각 프로세스를 별도의 run으로 추적합니다. 필요에 따라 W&B App UI에서 이들을 그룹으로 묶을 수 있습니다.
   * 모든 프로세스를 하나의 run에 기록합니다.

## 단일 프로세스 추적

이 섹션에서는 rank 0 프로세스에서만 확인 가능한 값과 메트릭 추적 방법을 설명합니다. 단일 프로세스에서 접근 가능한 메트릭만 추적할 경우에 적합한 방식입니다. 보통 GPU/CPU 사용량, 공유 검증 세트의 행동, 그레이디언트 및 파라미터, 대표적인 데이터 샘플의 손실 값 등이 해당합니다.

rank 0 프로세스 내부에서 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})로 W&B run을 초기화하고, [`wandb.log`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ko" >}})를 통해 실험을 기록합니다.

아래 [샘플 Python 스크립트 (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)는 PyTorch DDP와 두 개의 GPU를 사용해 단일 머신에서 메트릭을 추적하는 한 가지 방법을 보여줍니다. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn`의 `DistributedDataParallel`)는 분산 트레이닝에서 널리 쓰이는 라이브러리입니다. 기본 원리는 어떤 분산 트레이닝에도 동일하게 적용 가능하지만, 구현 방식은 다를 수 있습니다.

Python 스크립트의 흐름은 다음과 같습니다:
1. `torch.distributed.launch`로 여러 프로세스를 시작합니다.
2. `--local_rank` 커맨드라인 인수로 랭크를 확인합니다.
3. 만약 랭크가 0이면, [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 함수 내에서 조건부로 `wandb` 로그를 설정합니다.

```python
if __name__ == "__main__":
    # 인수 받아오기
    args = parse_args()

    if args.local_rank == 0:  # 메인 프로세스에서만 실행
        # wandb run 초기화
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP로 모델 트레이닝
        train(args, run)
    else:
        train(args)
```

[단일 프로세스에서 추적된 메트릭을 보여주는 예시 대시보드](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)를 확인해보세요.

이 대시보드에는 두 GPU의 시스템 메트릭(온도, 사용률 등)이 표시됩니다.

{{< img src="/images/track/distributed_training_method1.png" alt="GPU metrics dashboard" >}}

하지만, 에포크와 배치 크기별 손실 값은 오직 단일 GPU에서만 기록되었습니다.

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="Loss function plots" >}}

## 다중 프로세스 추적

W&B에서 여러 프로세스를 추적하려면 아래 방법 중 하나를 사용할 수 있습니다:
* 각 프로세스를 별도로 run으로 생성해 [각각 추적]({{< relref path="distributed-training/#track-each-process-separately" lang="ko" >}})
* [모든 프로세스를 하나의 run에 기록]({{< relref path="distributed-training/#track-all-processes-to-a-single-run" lang="ko" >}})

### 각 프로세스를 별도로 추적

이 섹션에서는 각 프로세스를 별도의 run으로 기록하는 방법을 다룹니다. 각 run에서는 해당 run의 메트릭, 아티팩트 등을 기록할 수 있습니다. 트레이닝이 끝난 뒤에는 `wandb.Run.finish()`를 호출해서 run이 완료됨을 표시하세요. 이를 통해 모든 프로세스가 정상적으로 종료될 수 있습니다.

여러 실험에서 run들을 구분하기 어려울 수 있으니, W&B를 초기화할 때 `group` 파라미터에 값을 부여하세요 (`wandb.init(group='group-name')`). 이를 통해 각각의 run이 어떤 실험에 속하는지 쉽게 추적할 수 있습니다. 실험 내 트레이닝 및 평가 W&B Runs를 관리하는 방법은 [Group Runs]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})에서 자세히 볼 수 있습니다.

{{% alert %}}
**개별 프로세스의 메트릭도 추적하려면 이 방법을 사용하세요.** 예시로는 각 노드의 데이터 및 예측값(데이터 분산 디버깅용), 메인 노드 외 배치별 메트릭 기록 등이 있습니다. 시스템 메트릭 전체 수집이나 메인 노드 요약 통계 추출에는 이 방법이 필수는 아닙니다.
{{% /alert %}}

아래는 W&B를 초기화할 때 group 파라미터를 설정하는 Python 코드 예시입니다:

```python
if __name__ == "__main__":
    # 인수 받아오기
    args = parse_args()
    # run 초기화
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # 실험의 모든 run을 하나의 그룹으로
    )
    # DDP로 모델 트레이닝
    train(args, run)

    run.finish()  # run이 완료됐음을 표시
```

W&B App UI에서 [다중 프로세스에서 추적된 메트릭의 예시 대시보드](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)를 확인해보세요. 왼쪽 사이드바에 두 개의 W&B Runs가 그룹으로 묶여있는 것을 볼 수 있습니다. 그룹을 클릭하면 해당 실험에 대한 별도의 그룹 페이지가 열리고, 각 프로세스에서 기록된 메트릭을 따로 확인할 수 있습니다.

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="Grouped distributed runs" >}}

위 이미지는 W&B App UI의 대시보드를 보여줍니다. 사이드바에는 두 개의 실험이 나타납니다. 하나는 ‘null’로, 나머지 하나(노란 테두리)는 ‘DPP’로 라벨링되어 있습니다. 그룹 확장(Group 드롭다운 선택) 시 해당 실험에 속한 W&B Runs를 확인할 수 있습니다.

### 모든 프로세스를 하나의 run으로 추적

{{% alert color="secondary"  %}}
`x_`로 시작하는 파라미터(예: `x_label`)는 퍼블릭 프리뷰 상태입니다. 피드백은 [W&B GitHub 저장소에 이슈](https://github.com/wandb/wandb)로 올려주세요.
{{% /alert %}}

{{% alert title="필수 조건" %}}
모든 프로세스를 하나의 run에 기록하려면 다음이 필요합니다:
- W&B Python SDK `v0.19.9` 이상

- W&B Server v0.68 이상
{{% /alert  %}}

이 방식에서는 주 노드(primary)와 하나 이상의 워커 노드(worker)가 존재합니다. 주 노드에서 W&B run을 초기화합니다. 각각의 워커 노드에서 주 노드의 run ID를 사용해 run을 초기화합니다. 트레이닝 도중 각 워커 노드는 동일한 run ID로 로그를 기록합니다. W&B는 모든 노드의 메트릭을 집계해 App UI에 표시합니다.

주 노드에서는 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})로 run을 초기화하고, 다음과 같이 `settings` 파라미터에 `wandb.Settings` 오브젝트를 전달합니다 (`wandb.init(settings=wandb.Settings()`) :

1. `mode` 파라미터는 `"shared"`로 설정해 공유 모드를 활성화
2. `[x_label](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L638)`에 고유 라벨 지정. 지정한 값은 W&B App UI의 로그와 시스템 메트릭에서 데이터가 어느 노드에서 왔는지 식별할 때 사용합니다. 지정하지 않으면 W&B가 hostname과 랜덤 해시로 생성합니다.
3. [`x_primary`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L660)를 `True`로 설정해 이 노드가 primary임을 표시
4. 옵션으로, W&B가 추적할 GPU 인덱스([0,1,2]) 목록을 `x_stats_gpu_device_ids`에 전달할 수 있습니다. 지정하지 않으면 모든 GPU의 메트릭이 추적됩니다.

주 노드의 run ID를 따로 기록해두세요. 각 워커 노드가 이 값으로 run을 초기화해야 합니다.

{{% alert %}}
`x_primary=True`는 primary 노드와 worker 노드를 구분합니다. primary 노드만이 설정 파일, 텔레메트리 등 노드간 공유되는 파일들을 업로드합니다. worker 노드는 이 파일들을 업로드하지 않습니다.
{{% /alert %}}

각 워커 노드에서는 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})로 run을 초기화할 때 아래처럼 설정하세요:
1. `settings` 파라미터로 `wandb.Settings` 오브젝트를 전달
   * `mode`를 `"shared"`로, `x_label`에 고유라벨, `x_primary`를 `False`로 설정
2. 주 노드에서 사용한 run ID를 `id` 파라미터에 전달
3. 옵션으로 [`x_update_finish_state`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L772)를 `False`로 설정. 이를 통해 primary가 아닌 노드가 [run의 state]({{< relref path="/guides/models/track/runs/#run-states" lang="ko" >}})를 미리 `finished`로 바꾸는 것을 막아, run의 상태가 일관되고 primary가 관리하게 할 수 있습니다.

{{% alert %}}
주 노드의 run ID는 환경 변수로 설정해 각 워커 노드가 사용할 수 있도록 관리하는 방식을 추천합니다.
{{% /alert %}}

아래 샘플 코드는 다중 프로세스를 하나의 run에 기록할 때 필요한 기본 개념만을 보여줍니다:

```python
import wandb

# primary 노드에서 run 초기화
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="rank_0", 
        mode="shared", 
        x_primary=True,
        x_stats_gpu_device_ids=[0, 1],  # (옵션) GPU 0, 1만 추적
        )
)

# primary 노드의 run ID를 기록
# 각 워커 노드는 이 ID가 필요
run_id = run.id

# 워커 노드에서 primary run ID로 run 초기화
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared", x_primary=False),
	id=run_id,
)

# 워커 노드에서 primary run ID로 run 초기화
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared", x_primary=False),
	id=run_id,
)
```

실제 환경에서는 각각의 워커 노드가 별도의 머신에서 구동될 수 있습니다.

{{% alert %}}
[GKE의 멀티 노드, 멀티 GPU Kubernetes 클러스터에서 모델을 트레이닝하는 엔드투엔드 예시](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA)를 참고하세요.
{{% /alert %}}

실행 로그(console logs)를 여러 노드 프로세스별로 확인하려면 아래를 따라주세요:

1. 해당 run이 속한 프로젝트로 이동합니다.
2. 왼쪽 사이드바에서 **Runs** 탭을 클릭합니다.
3. 원하는 run을 클릭합니다.
4. 왼쪽 사이드바의 **Logs** 탭을 클릭합니다.

UI의 콘솔 로그 페이지 상단에 있는 검색창에서 `x_label`에 지정한 값으로 필터링할 수 있습니다. 아래 이미지는 `rank0`, `rank1`, `rank2`, `rank3`, `rank4`, `rank5`, `rank6` 값을 `x_label`에 지정했을 때 필터 메뉴를 보여줍니다.

{{< img src="/images/track/multi_node_console_logs.png" alt="Multi-node console logs" >}}

더 자세한 내용은 [콘솔 로그 문서]({{< relref path="/guides/models/app/console-logs/" lang="ko" >}})를 참고하세요.

W&B는 모든 노드에서 시스템 메트릭을 집계해 W&B App UI에 표시합니다. 아래 이미지는 여러 노드(각각 `x_label`에 따라 `rank_0`, `rank_1`, `rank_2`로 식별)의 시스템 메트릭을 예시로 보여줍니다.

{{< img src="/images/track/multi_node_system_metrics.png" alt="Multi-node system metrics" >}}

라인 플롯 커스터마이즈 등 세부 정보는 [Line plots]({{< relref path="/guides/models/app/features/panels/line-plot/" lang="ko" >}}) 문서를 참고하세요.

## 예시 유스 케이스

아래 코드조각은 고급 분산 유스 케이스에서 자주 등장하는 상황을 보여줍니다.

### 프로세스 스폰

스폰된 프로세스에서 run을 시작해야 한다면 main 함수에서 `wandb.setup()` 메소드를 사용하세요:

```python
import multiprocessing as mp

def do_work(n):
    with wandb.init(config=dict(n=n)) as run:
        run.log(dict(this=n * n))

def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

### run 공유

프로세스 간에 run 오브젝트를 인수로 전달해 run을 공유할 수도 있습니다:

```python
def do_work(run):
    with wandb.init() as run:
        run.log(dict(this=1))

def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()
    run.finish()  # run이 완료됐음을 표시


if __name__ == "__main__":
    main()
```

W&B는 로그 기록 순서를 보장하지 않습니다. 동기화는 사용자 코드에서 직접 관리해 주셔야 합니다.

## 문제 해결

W&B와 분산 트레이닝을 함께 사용할 때 발생할 수 있는 대표적인 이슈는 두 가지입니다:

1. **트레이닝 시작 시 멈춤** - 분산 트레이닝의 멀티프로세싱과 `wandb` 프로세스가 충돌하면 `wandb` 프로세스가 멈출 수 있습니다.
2. **트레이닝 종료 시 멈춤** - 트레이닝이 끝난 후 `wandb`가 언제 종료해야 하는지 몰라서 잡이 멈출 수 있습니다. 마지막에 `wandb.Run.finish()` API를 반드시 호출해 run의 종료를 W&B에 알려주세요. 이 API가 데이터를 업로드하고 W&B 종료를 처리합니다.

W&B에서는 분산 잡의 안정성을 위해 `wandb service` 커맨드 사용을 권장합니다. 위 문제들은 보통 wandb service 지원 전 SDK 버전에서 자주 발견됩니다.

### W&B Service 활성화

W&B SDK 버전에 따라 W&B Service가 이미 기본적으로 활성화되어 있을 수 있습니다.

#### W&B SDK 0.13.0 이상

W&B SDK `0.13.0` 이상에서는 기본적으로 W&B Service가 활성화되어 있습니다.

#### W&B SDK 0.12.5 이상

W&B SDK 버전 0.12.5 이상에서 W&B Service를 활성화하려면 Python 스크립트 main 함수에서 `wandb.require` 메소드에 `"service"` 문자열을 전달하세요:

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # 이 아래에 나머지 스크립트를 작성하면 됩니다
```

최적의 경험을 위해 최신 버전으로 업그레이드할 것을 권장합니다.

**W&B SDK 0.12.4 이하**

W&B SDK 0.12.4 이하를 사용한다면, 대신 환경 변수 `WANDB_START_METHOD`를 `"thread"`로 설정해 멀티스레딩을 사용하세요.