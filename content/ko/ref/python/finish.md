---
title: finish
menu:
  reference:
    identifier: ko-ref-python-finish
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_run.py#L4109-L4130 >}}

run 을 종료하고 남아있는 모든 데이터를 업로드합니다.

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

W&B run 의 완료를 표시하고 모든 데이터가 서버 와 동기화되도록 합니다. run 의 최종 상태는 종료 조건 및 동기화 상태에 따라 결정됩니다.

#### run 상태:

- Running: 데이터를 로깅하거나 하트비트를 보내는 활성 run 입니다.
- Crashed: 예기치 않게 하트비트 전송을 중단한 run 입니다.
- Finished: 모든 데이터가 동기화되어 성공적으로 완료된 run 입니다( `exit_code=0` ).
- Failed: 오류와 함께 완료된 run 입니다( `exit_code!=0` ).

| Args |  |
| :--- | :--- |
|  `exit_code` | run 의 종료 상태를 나타내는 정수입니다. 성공 시 0을 사용하고, 다른 값은 run 이 실패했음을 표시합니다. |
|  `quiet` | 더 이상 사용되지 않습니다. `wandb.Settings(quiet=...)` 를 사용하여 로깅 verbosity를 구성하세요. |
