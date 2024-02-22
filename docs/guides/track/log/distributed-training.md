---
description: Use W&B to log distributed training experiments with multiple GPUs.
displayed_sidebar: default
---

# 분산 학습 실험 기록하기

<head>
  <title>분산 학습 실험 기록하기</title>
</head>


분산 학습에서는 여러 GPU를 병렬로 사용하여 모델을 학습합니다. W&B는 분산 학습 실험을 추적하기 위해 두 가지 패턴을 지원합니다:

1. **하나의 프로세스**: 하나의 프로세스에서 W&B([`wandb.init`](../../../ref//python/init.md))를 초기화하고 실험을 기록([`wandb.log`](../../../ref//python/log.md))합니다. 이는 [PyTorch 분산 데이터 병렬](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) 클래스와 함께 분산 학습 실험을 기록하기 위한 일반적인 해결책입니다. 일부 경우에서는, 사용자가 다른 프로세스에서 데이터를 멀티프로세싱 큐(또는 다른 통신 원시)를 통해 메인 로깅 프로세스로 전달합니다.
2. **여러 프로세스**: 모든 프로세스에서 W&B([`wandb.init`](../../../ref//python/init.md))를 초기화하고 실험을 기록([`wandb.log`](../../../ref//python/log.md))합니다. 각 프로세스는 사실상 별개의 실험입니다. W&B를 초기화할 때 `group` 파라미터를 사용하여 공유된 실험을 정의하고 W&B 앱 UI에서 로그된 값을 함께 그룹화합니다(`wandb.init(group='group-name')`).

다음 예제들은 단일 기계에서 두 개의 GPU를 사용하여 PyTorch DDP를 사용하여 W&B로 메트릭을 추적하는 방법을 보여줍니다. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`torch.nn`의 `DistributedDataParallel`)는 분산 학습을 위한 인기 있는 라이브러리입니다. 기본 원리는 모든 분산 학습 설정에 적용되지만, 구현의 세부 사항은 다를 수 있습니다.

:::info
이 예제들 뒤에 있는 코드를 W&B GitHub 예제 저장소 [여기](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp)에서 탐색하세요. 특히, 하나의 프로세스와 여러 프로세스 방법을 구현하는 방법에 대한 정보는 [`log-ddp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python 스크립트를 확인하세요.
:::

### 방법 1: 하나의 프로세스

이 방법에서는 rank 0 프로세스만 추적합니다. 이 방법을 구현하려면 W&B(`wandb.init`)를 초기화하고, W&B 실행을 시작하며, rank 0 프로세스 내에서 메트릭을 기록(`wandb.log`)합니다. 이 방법은 간단하고 견고하지만, 다른 프로세스(예: 손실 값 또는 배치에서의 입력)의 모델 메트릭을 기록하지 않습니다. 시스템 메트릭(예: 사용량 및 메모리)은 모든 프로세스에서 이 정보가 사용 가능하므로 모든 GPU에 대해 여전히 기록됩니다.

:::info
**단일 프로세스에서 사용 가능한 메트릭만 추적하려면 이 방법을 사용하세요**. 일반적인 예로는 GPU/CPU 사용량, 공유 검증 세트에서의 동작, 그레이디언트 및 파라미터, 대표 데이터 예시에서의 손실 값이 있습니다.
:::

[샘플 Python 스크립트(`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py)에서는 rank가 0인지 확인합니다. 이를 위해, 우선 `torch.distributed.launch`로 여러 프로세스를 시작합니다. 다음으로, `--local_rank` 명령줄 인수로 rank를 확인합니다. rank가 0으로 설정된 경우, [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) 함수 내에서 조건부로 `wandb` 로깅을 설정합니다. Python 스크립트 내에서 다음과 같은 확인을 사용합니다:

```python showLineNumbers
if __name__ == "__main__":
    # 인수 가져오기
    args = parse_args()

    if args.local_rank == 0:  # 메인 프로세스에서만
        # wandb 실행 초기화
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # DDP로 모델 학습
        train(args, run)
    else:
        train(args)
```

단일 프로세스에서 추적된 메트릭의 [예제 대시보드](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system)를 W&B 앱 UI에서 탐색해 보세요. 대시보드는 두 GPU 모두에 대해 추적된 온도 및 사용량과 같은 시스템 메트릭을 표시합니다.

![](/images/track/distributed_training_method1.png)

그러나, 손실 값은 에포크 및 배치 크기 함수로 단일 GPU에서만 기록되었습니다.

![](/images/experiments/loss_function_single_gpu.png)

### 방법 2: 여러 프로세스

이 방법에서는 작업의 각 프로세스를 추적하여, 각 프로세스에서 별도로 `wandb.init()` 및 `wandb.log()`를 호출합니다. 학습이 끝날 때 `wandb.finish()`를 호출하여 실행이 완료되었음을 표시하고 모든 프로세스가 제대로 종료되도록 하는 것이 좋습니다.

이 방법은 로깅을 위해 더 많은 정보를 사용할 수 있게 합니다. 그러나, W&B 앱 UI에서 여러 W&B 실행을 보고하는 것은 어려울 수 있습니다. 이를 완화하기 위해, W&B를 초기화할 때 그룹 파라미터에 값을 제공하여 주어진 실험에 속한 W&B 실행을 추적할 수 있습니다. 실험에서 학습 및 평가 W&B 실행을 추적하는 방법에 대한 자세한 내용은 [그룹 실행](../../runs/grouping.md)을 참조하세요.

:::info
**개별 프로세스에서 메트릭을 추적하려면 이 방법을 사용하세요**. 일반적인 예로는 각 노드에서의 데이터 및 예측값(데이터 분배 디버깅을 위해) 및 메인 노드 외부의 개별 배치에서 메트릭이 있습니다. 이 방법은 모든 노드에서 시스템 메트릭을 얻거나 메인 노드에서 사용 가능한 요약 통계를 얻기 위해 필요하지 않습니다.
:::

다음 Python 코드 조각은 W&B를 초기화할 때 그룹 파라미터를 설정하는 방법을 보여줍니다:

```python
if __name__ == "__main__":
    # 인수 가져오기
    args = parse_args()
    # 실행 초기화
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # 실험의 모든 실행을 하나의 그룹으로
    )
    # DDP로 모델 학습
    train(args, run)
```

여러 프로세스에서 추적된 메트릭의 [예제 대시보드](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna)를 W&B 앱 UI에서 탐색해 보세요. 왼쪽 사이드바에서 두 개의 W&B 실행이 함께 그룹화된 것을 주목하세요. 그룹을 클릭하여 실험에 대한 전용 그룹 페이지를 봅니다. 전용 그룹 페이지는 각 프로세스에서 별도로 메트릭을 표시합니다.

![](/images/experiments/dashboard_grouped_runs.png)

앞의 이미지는 W&B 앱 UI 대시보드를 보여줍니다. 사이드바에서 'null'이라고 표시된 실험 하나와 노란색 상자로 묶인 'DPP'라고 불리는 두 번째 실험을 볼 수 있습니다. 그룹을 확장하면(그룹 드롭다운 선택) 해당 실험에 연관된 W&B 실행을 볼 수 있습니다.

### 분산 학습 문제를 피하기 위해 W&B 서비스 사용

W&B와 분산 학습을 사용할 때 마주칠 수 있는 두 가지 일반적인 문제가 있습니다:

1. **학습 시작 시 멈춤** - 분산 학습의 멀티프로세싱과 `wandb` 멀티프로세싱이 상호 작용할 때 `wandb` 프로세스가 멈출 수 있습니다.
2. **학습 끝날 때 멈춤** - `wandb` 프로세스가 언제 종료해야 하는지 모를 때 학습 작업이 멈출 수 있습니다. Python 스크립트의 끝에서 `wandb.finish()` API를 호출하여 W&B에 실행이 끝났음을 알리세요. wandb.finish() API는 데이터 업로드를 완료하고 W&B가 종료되도록 합니다.

분산 작업의 신뢰성을 향상시키기 위해 `wandb service` 사용을 권장합니다. 앞서 언급한 학습 문제는 wandb service를 사용할 수 없는 W&B SDK 버전에서 일반적으로 발견됩니다.

### W&B 서비스 활성화

W&B SDK의 버전에 따라 W&B 서비스가 기본적으로 활성화되어 있을 수 있습니다.

#### W&B SDK 0.13.0 이상

W&B 서비스는 W&B SDK `0.13.0` 버전 이상에서 기본적으로 활성화됩니다.

#### W&B SDK 0.12.5 이상

W&B SDK 버전 0.12.5 이상의 경우 Python 스크립트를 수정하여 W&B 서비스를 활성화하세요. 메인 함수 내에서 `wandb.require` 메서드를 사용하고 `"service"` 문자열을 전달하세요:

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # 여기에 나머지 스크립트
```

최적의 경험을 위해 최신 버전으로 업그레이드하는 것이 좋습니다.

**W&B SDK 0.12.4 이하**

W&B SDK 버전 0.12.4 이하를 사용하는 경우 `WANDB_START_METHOD` 환경 변수를 `"thread"`로 설정하여 멀티스레딩을 사용하세요.

### 멀티프로세싱에 대한 예제 사용 사례

다음 코드 조각은 고급 분산 사용 사례에 대한 일반적인 방법을 보여줍니다.

#### 프로세스 생성

스폰된 프로세스에서 W&B 실행을 시작하는 경우 메인 함수에서 `wandb.setup()[line 8]`메서드를 사용하세요:

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

#### W&B 실행 공유

W&B 실행 개체를 인수로 전달하여 프로세스 간에 W&B 실행을 공유하세요:

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
로그 순서를 보장할 수 없습니다. 동기화는 스크립트의 저자가 수행해야 합니다.
:::