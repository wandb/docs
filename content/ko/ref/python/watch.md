---
title: watch
menu:
  reference:
    identifier: ko-ref-python-watch
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L2801-L2836 >}}

주어진 PyTorch 모델에 훅을 연결하여 그레이디언트와 모델의 계산 그래프를 모니터링합니다.

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

이 함수는 트레이닝 중 파라미터, 그레이디언트 또는 둘 다를 추적할 수 있습니다. 앞으로 임의의 기계 학습 모델을 지원하도록 확장되어야 합니다.

| Args |  |
| :--- | :--- |
|  models (Union[torch.nn.Module, Sequence[torch.nn.Module]]): 모니터링할 단일 모델 또는 모델의 시퀀스입니다. criterion (Optional[torch.F]): 최적화할 손실 함수 (선택 사항). log (Optional[Literal["gradients", "parameters", "all"]]): "gradients", "parameters" 또는 "all"을 로깅할지 여부를 지정합니다. 로깅을 비활성화하려면 None으로 설정합니다. (기본값="gradients") log_freq (int): 그레이디언트 및 파라미터를 로깅할 빈도 (배치 단위). (기본값=1000) idx (Optional[int]): `wandb.watch`로 여러 모델을 추적할 때 사용되는 인덱스입니다. (기본값=None) log_graph (bool): 모델의 계산 그래프를 로깅할지 여부입니다. (기본값=False) |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  `wandb.init`이 호출되지 않았거나 모델 중 하나라도 `torch.nn.Module`의 인스턴스가 아닌 경우. |
