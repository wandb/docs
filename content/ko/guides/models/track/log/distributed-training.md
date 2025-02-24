---
title: Log distributed training experiments
description: W&B를 사용하여 여러 개의 GPU를 사용하는 분산 트레이닝 Experiments을 로그합니다.
menu:
  default:
    identifier: ko-guides-models-track-log-distributed-training
    parent: log-objects-and-media
---

분산 트레이닝에서는 여러 GPU를 병렬로 사용하여 모델을 트레이닝합니다. W&B는 분산 트레이닝 Experiments를 추적하기 위한 두 가지 패턴을 지원합니다.

1. **단일 프로세스**: 단일 프로세스에서 W&B ([`wandb.init`]({{< relref path="/ref//python/init.md" lang="ko" >}}))를 초기화하고 Experiments ([`wandb.log`]({{< relref path="/ref//python/log.md" lang="ko" >}}))를 기록합니다. 이는 [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) 클래스로 분산 트레이닝 Experiments를 로깅하는 일반적인 솔루션입니다. 경우에 따라 사용자는 다중 처리 대기열(또는 다른 통신 기본 요소)을 사용하여 다른 프로세스의 데이터를 기본 로깅 프로세스로 보냅니다.
2. **다중 프로세스**: 모든 프로세스에서 W&B ([`wandb.init`]({{< relref path="/ref//python/init.md" lang="ko" >}}))를 초기화하고 Experiments ([`wandb.log`]({{< relref path="/ref//python/log.md" >}}))를 기록합니다. 각 프로세스는 사실상 별도의 experiment입니다. W&B를 초기화할 때 `group` 파라미터(`wandb.init(group='group-name')`)를 사용하여 공유 experiment를 정의하고 W&B App UI에서 기록된 값들을 함께 그룹화합니다.

다음 예제에서는 단일 머신에서 두 개의 GPU를 사용하여 PyTorch DDP로 W&B를 통해 메트릭을 추적하는 방법을 보여줍니다. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn`의 `DistributedDataParallel`)은 분산 트레이닝을 위한 널리 사용되는 라이브러리입니다. 기본 원칙은 모든 분산 트레이닝 설정에 적용되지만 구현 세부 사항은 다를 수 있습니다.

{{% alert %}}
W&B GitHub 예제 리포지토리 [여기](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)에서 이러한 예제에 대한 코드를 살펴보세요. 특히, 단일 프로세스 및 다중 프로세스 메소드를 구현하는 방법에 대한 정보는 [`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python 스크립트를 참조하세요.
{{% /alert %}}

### 방법 1: 단일 프로세스

이 방법에서는 순위 0 프로세스만 추적합니다. 이 방법을 구현하려면 W&B(`wandb.init`)를 초기화하고, W&B Run을 시작하고, 순위 0 프로세스 내에서 메트릭(`wandb.log`)을 기록합니다. 이 방법은 간단하고 강력하지만 다른 프로세스의 모델 메트릭(예: 배치로부터의 손실 값 또는 입력)은 기록하지 않습니다. 사용량 및 메모리와 같은 시스템 메트릭은 해당 정보를 모든 프로세스에서 사용할 수 있으므로 모든 GPU에 대해 계속 기록됩니다.

{{% alert %}}
**단일 프로세스에서 사용할 수 있는 메트릭만 추적하려면 이 방법을 사용하세요**. 일반적인 예로는 GPU/CPU 사용률, 공유 검증 세트의 행동, 그래디언트 및 파라미터, 대표적인 데이터 예제의 손실 값이 있습니다.
{{% /alert %}}

[샘플 Python 스크립트 (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) 내에서 순위가 0인지 확인합니다. 이를 구현하기 위해 먼저 `torch.distributed.launch`로 여러 프로세스를 시작합니다. 다음으로 `--local_rank` 커맨드라인 인수로 순위를 확인합니다. 순위가 0으로 설정된 경우 [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 함수에서 조건부로 `wandb` 로깅을 설정합니다. Python 스크립트 내에서 다음 검사를 사용합니다.

```python showLineNumbers
if __name__ == "__main__":
    # Get args
    args = parse_args()

    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # Train model with DDP
        train(args, run)
    else:
        train(args)
```

W&B App UI를 탐색하여 단일 프로세스에서 추적된 메트릭의 [예제 대시보드](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)를 확인하세요. 대시보드는 두 GPU에 대해 추적된 온도 및 사용률과 같은 시스템 메트릭을 표시합니다.

{{< img src="/images/track/distributed_training_method1.png" alt="" >}}

그러나 에포크 및 배치 크기 함수로서의 손실 값은 단일 GPU에서만 기록되었습니다.

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="" >}}

### 방법 2: 다중 프로세스

이 방법에서는 작업의 각 프로세스를 추적하고 각 프로세스에서 `wandb.init()` 및 `wandb.log()`를 호출합니다. 모든 프로세스가 제대로 종료되도록 트레이닝이 끝나면 `wandb.finish()`를 호출하여 Run이 완료되었음을 표시하는 것이 좋습니다.

이 방법을 사용하면 로깅을 위해 더 많은 정보에 액세스할 수 있습니다. 그러나 여러 W&B Runs가 W&B App UI에 보고된다는 점에 유의하세요. 여러 Experiments에서 W&B Runs를 추적하기 어려울 수 있습니다. 이를 완화하려면 W&B를 초기화할 때 group 파라미터에 값을 제공하여 지정된 experiment에 속하는 W&B Run을 추적하세요. Experiments에서 트레이닝 및 평가 W&B Runs를 추적하는 방법에 대한 자세한 내용은 [Run 그룹화]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})를 참조하세요.

{{% alert %}}
**개별 프로세스의 메트릭을 추적하려는 경우 이 방법을 사용하세요**. 일반적인 예로는 각 노드의 데이터 및 예측(데이터 배포 디버깅용)과 기본 노드 외부의 개별 배치에 대한 메트릭이 있습니다. 이 방법은 모든 노드에서 시스템 메트릭을 가져오거나 기본 노드에서 사용할 수 있는 요약 통계를 가져오는 데 필요하지 않습니다.
{{% /alert %}}

다음 Python 코드 조각은 W&B를 초기화할 때 group 파라미터를 설정하는 방법을 보여줍니다.

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # all runs for the experiment in one group
    )
    # Train model with DDP
    train(args, run)
```

W&B App UI를 탐색하여 다중 프로세스에서 추적된 메트릭의 [예제 대시보드](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)를 확인하세요. 왼쪽 사이드바에 함께 그룹화된 두 개의 W&B Runs가 있습니다. 그룹을 클릭하여 experiment의 전용 그룹 페이지를 확인하세요. 전용 그룹 페이지에는 각 프로세스의 메트릭이 개별적으로 표시됩니다.

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="" >}}

위의 이미지는 W&B App UI 대시보드를 보여줍니다. 사이드바에는 두 개의 Experiments가 있습니다. 하나는 'null'로 레이블이 지정되고 두 번째는 'DPP'(노란색 상자로 묶임)라고 합니다. 그룹을 확장하면(그룹 드롭다운 선택) 해당 experiment와 연결된 W&B Runs가 표시됩니다.

### W&B Service를 사용하여 일반적인 분산 트레이닝 문제 방지

W&B 및 분산 트레이닝을 사용할 때 발생할 수 있는 두 가지 일반적인 문제가 있습니다.

1. **트레이닝 시작 시 멈춤** - `wandb` 다중 처리가 분산 트레이닝의 다중 처리를 방해하는 경우 `wandb` 프로세스가 멈출 수 있습니다.
2. **트레이닝 종료 시 멈춤** - `wandb` 프로세스가 종료해야 할 시점을 모르는 경우 트레이닝 작업이 멈출 수 있습니다. Python 스크립트가 끝나면 `wandb.finish()` API를 호출하여 W&B에 Run이 완료되었음을 알립니다. wandb.finish() API는 데이터 업로드를 완료하고 W&B를 종료합니다.

`wandb service`를 사용하여 분산 작업의 안정성을 향상시키는 것이 좋습니다. 앞서 언급한 두 가지 트레이닝 문제는 wandb service를 사용할 수 없는 W&B SDK 버전에서 흔히 발생합니다.

### W&B Service 활성화

W&B SDK 버전에 따라 W&B Service가 기본적으로 활성화되어 있을 수 있습니다.

#### W&B SDK 0.13.0 이상

W&B SDK 버전 `0.13.0` 이상의 경우 W&B Service가 기본적으로 활성화되어 있습니다.

#### W&B SDK 0.12.5 이상

Python 스크립트를 수정하여 W&B SDK 버전 0.12.5 이상에 대해 W&B Service를 활성화합니다. `wandb.require` 메소드를 사용하고 기본 함수 내에서 문자열 `"service"`를 전달합니다.

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

최적의 경험을 위해 최신 버전으로 업그레이드하는 것이 좋습니다.

**W&B SDK 0.12.4 이하**

W&B SDK 버전 0.12.4 이하를 사용하는 경우 `WANDB_START_METHOD` 환경 변수를 `"thread"`로 설정하여 대신 멀티스레딩을 사용하세요.

### 다중 처리에 대한 예제 유스 케이스

다음 코드 조각은 고급 분산 유스 케이스에 대한 일반적인 방법을 보여줍니다.

#### Spawn 프로세스

스폰된 프로세스에서 W&B Run을 시작하는 경우 기본 함수에서 `wandb.setup()[line 8]` 메소드를 사용합니다.

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

W&B Run 오브젝트를 인수로 전달하여 프로세스 간에 W&B Runs를 공유합니다.

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


{{% alert %}}
로깅 순서를 보장할 수 없습니다. 동기화는 스크립트 작성자가 수행해야 합니다.
{{% /alert %}}
