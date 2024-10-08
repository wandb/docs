---
title: Log distributed training experiments
description: W&B를 사용하여 여러 GPU로 분산 트레이닝 실험을 로그합니다.
displayed_sidebar: default
---

분산 트레이닝에서는 여러 GPU를 사용하여 모델을 병렬로 학습합니다. W&B는 분산 트레이닝 실험을 추적하기 위해 두 가지 패턴을 지원합니다:

1. **단일 프로세스**: 단일 프로세스에서 W&B를 초기화([`wandb.init`](../../../ref//python/init.md))하고 실험을 로그합니다([`wandb.log`](../../../ref//python/log.md)). 이는 [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) 클래스와 함께 분산 트레이닝 실험을 로그하는 일반적인 솔루션입니다. 일부 경우, 사용자는 멀티프로세싱 큐(또는 다른 통신 원시 도구)를 사용하여 다른 프로세스에서 데이터를 메인 로그 프로세스로 전송합니다.
2. **다중 프로세스**: 각 프로세스에서 W&B를 초기화([`wandb.init`](../../../ref//python/init.md))하고 실험을 로그합니다([`wandb.log`](../../../ref//python/log.md)). 각 프로세스는 실질적으로 별도의 실험입니다. W&B를 초기화할 때 `group` 파라미터를 사용하여(`wandb.init(group='group-name')`) 공유된 실험을 정의하고 W&B 앱 UI에서 로그된 값을 함께 그룹화합니다.

다음 예제들은 단일 머신에서 두 개의 GPU를 사용하여 PyTorch DDP와 함께 W&B를 사용하여 메트릭을 추적하는 방법을 보여줍니다. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn`의 `DistributedDataParallel`)는 분산 트레이닝을 위한 인기 있는 라이브러리입니다. 기본 원칙은 모든 분산 트레이닝 설정에 적용됩니다. 하지만 구현의 세부 사항은 다를 수 있습니다.

:::info
이 예제들의 코드 분석은 W&B GitHub 예제 저장소에서 [여기](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)에서 확인할 수 있습니다. 특히 하나의 프로세스 및 여러 프로세스 방법을 구현하는 방법에 대한 정보는 [`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python 스크립트를 참조하십시오.
:::

### 메소드 1: 단일 프로세스

이 방법에서는 순위 0 프로세스만 추적합니다. 이 방법을 구현하려면, W&B를 초기화(`wandb.init`)하고 W&B Run을 시작하여 순위 0 프로세스 내에서 메트릭을 로그합니다(`wandb.log`). 이 방법은 간단하고 견고하지만, 다른 프로세스에서 모델 메트릭(예: 손실 값 또는 배치의 입력값)을 로그하지 않습니다. 시스템 메트릭(예: 사용량 및 메모리)은 모든 GPU에 대해 여전히 로그됩니다. 이러한 정보는 모든 프로세스에서 사용할 수 있기 때문입니다.

:::info
**단일 프로세스에서 사용할 수 있는 메트릭만 추적하려면 이 방법을 사용하세요**. 일반적인 예로는 GPU/CPU 사용량, 공유 검증 세트의 행동, 그레이디언트 및 파라미터, 대표 데이터 예제의 손실 값 등이 있습니다.
:::

[샘플 Python 스크립트 (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)에서 순위가 0인지 확인합니다. 이를 위해 먼저 `torch.distributed.launch`를 사용하여 여러 프로세스를 시작합니다. 다음으로, `--local_rank` 커맨드라인 인수를 사용하여 순위를 확인합니다. 순위가 0으로 설정된 경우, `wandb` 로그를 [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 함수에서 조건부로 설정합니다. Python 스크립트 내에서는 다음과 같은 확인을 사용합니다:

```python showLineNumbers
if __name__ == "__main__":
    # 인수 가져오기
    args = parse_args()

    if args.local_rank == 0:  # 메인 프로세스에서만
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

W&B 앱 UI를 탐색하여 단일 프로세스에서 추적한 메트릭의 [예시 대시보드](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)를 확인하세요. 대시보드는 양쪽 GPU에 대해 추적된 온도와 사용량 같은 시스템 메트릭을 표시합니다.

![](/images/track/distributed_training_method1.png)

그러나 에포크와 배치 크기의 함수로서 손실 값은 단일 GPU에서만 로그되었습니다.

![](/images/experiments/loss_function_single_gpu.png)

### 메소드 2: 다중 프로세스

이 방법에서는 작업 내의 각 프로세스를 추적하고 각각의 프로세스에서 별도로 `wandb.init()` 및 `wandb.log()`를 호출합니다. 트레이닝이 종료되었음을 표시하고 모든 프로세스가 올바르게 종료되도록 하려면 트레이닝 마지막에 `wandb.finish()`를 호출하는 것을 권장합니다.

이 방법은 로그를 위한 더 많은 정보를 제공합니다. 하지만 W&B 앱 UI에서는 여러 W&B Runs가 보고됩니다. 여러 실험 간의 W&B Runs를 추적하기 어려울 수 있습니다. 이를 완화하기 위해, W&B를 초기화할 때 group 파라미터에 값을 제공하여 어느 W&B Run이 특정 실험에 속하는지를 추적하십시오. 실험에서 트레이닝 및 평가 W&B Runs를 추적하는 방법에 대해 더 알고 싶다면 [Group Runs](../../runs/grouping.md)를 참조하세요.

:::info
**각 프로세스의 메트릭을 추적하려면 이 방법을 사용하세요**. 일반적인 예로는 각 노드에서의 데이터 및 예측값(데이터 분포 디버깅용)과, 메인 노드 외부의 개별 배치에서의 메트릭이 있습니다. 이 방법은 모든 노드에서 시스템 메트릭을 얻거나, 메인 노드에서 사용 가능한 요약 통계를 얻기 위해 필요하지 않습니다.
:::

W&B를 초기화할 때 group 파라미터를 설정하는 방법을 보여주는 다음 Python 코드조각을 확인하십시오:

```python
if __name__ == "__main__":
    # 인수 가져오기
    args = parse_args()
    # run 초기화
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # 실험의 모든 runs를 하나의 그룹에 저장
    )
    # DDP로 모델 트레이닝
    train(args, run)
```

W&B 앱 UI를 탐색하여 여러 프로세스에서 추적한 메트릭의 [예시 대시보드](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)를 확인하세요. 왼쪽 사이드바에는 두 개의 W&B Runs가 그룹화되어 있음을 주목하세요. 그룹을 클릭하여 실험에 대한 전용 그룹 페이지를 확인하세요. 전용 그룹 페이지는 각 프로세스의 메트릭을 개별적으로 표시합니다.

![](/images/experiments/dashboard_grouped_runs.png)

이전 이미지는 W&B 앱 UI 대시보드를 보여줍니다. 사이드바에는 두 개의 실험이 표시됩니다. 하나는 'null'로 레이블이 붙어있고, 다른 하나(노란색 상자로 둘러싸인)는 'DPP'로 이름 붙어 있습니다. 그룹을 확장(그룹 드롭다운 선택)하면 해당 실험과 관련된 W&B Runs를 볼 수 있습니다.

### 일반적인 분산 트레이닝 문제를 피하기 위해 W&B 서비스를 사용하세요.

W&B와 분산 트레이닝을 사용할 때 겪을 수 있는 두 가지 일반적인 문제는 다음과 같습니다:

1. **트레이닝 시작 시 멈춤** - `wandb` 프로세스가 `wandb` 멀티프로세싱이 분산 트레이닝의 멀티프로세싱과 간섭할 경우 멈출 수 있습니다.
2. **트레이닝 종료 시 멈춤** - `wandb` 프로세스가 종료할 때를 알지 못하면 트레이닝 작업이 멈출 수 있습니다. Python 스크립트의 끝에서 `wandb.finish()` API를 호출하여 W&B가 Run이 완료되었음을 알 수 있도록 하세요. `wandb.finish()` API는 데이터를 업로드하는 작업을 끝내고 W&B를 종료하게 할 것입니다.

분산 작업의 신뢰성을 높이기 위해 `wandb 서비스`를 사용하는 것을 권장합니다. 앞서 언급한 트레이닝 문제들은 주로 W&B 서비스가 없는 W&B SDK 버전에서 흔히 발생합니다.

### W&B 서비스 활성화

사용 중인 W&B SDK 버전에 따라 기본적으로 W&B 서비스가 활성화되어 있을 수 있습니다.

#### W&B SDK 0.13.0 이상

W&B 서비스는 W&B SDK 버전 `0.13.0` 및 이상에서 기본적으로 활성화됩니다.

#### W&B SDK 0.12.5 이상

W&B SDK 버전 0.12.5 이상에 대해 W&B 서비스를 활성화하려면 Python 스크립트를 수정하세요. 메인 함수 내에서 `wandb.require` 메소드를 사용하고 문자열 `"service"`를 전달하세요:

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # 여기에 나머지 스크립트 추가
```

최적의 경험을 위해 최신 버전으로 업그레이드하는 것을 권장합니다.

**W&B SDK 0.12.4 이하**

W&B SDK 버전 0.12.4 이하를 사용하는 경우 대신 멀티스레딩을 사용하려면 `WANDB_START_METHOD` 환경 변수를 `"thread"`로 설정하세요.

### 멀티프로세싱에 대한 예제 유스 케이스

다음 코드조각은 고급 분산 유스 케이스에 대한 일반적인 메소드를 보여줍니다.

#### 프로세스 생성

스폰된 프로세스에서 W&B Run을 시작하는 경우 메인 함수에서 `wandb.setup()`[8행] 메소드를 사용하세요:

```python showLineNumbers
import multiprocessing as mp


def do_work(n):
    run = wandb.init(config=dict(n=n))
    run.log(dict(this=n * n))


def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

#### W&B Run 공유

W&B Run 오브젝트를 인수로 전달하여 프로세스 간에 W&B Runs를 공유하세요:

```python showLineNumbers
def do_work(run):
    run.log(dict(this=1))


def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()


if __name__ == "__main__":
    main()
```

:::info
로그 순서를 보장할 수 없음을 유념하십시오. 동기화는 스크립트 작성자가 해야 합니다.
:::