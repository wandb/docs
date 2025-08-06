---
title: 'finish()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-finish
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>함수</kbd> `finish`

```python
finish(exit_code: 'int | None' = None, quiet: 'bool | None' = None) → None
```

run 을 종료하고 남아 있는 데이터를 업로드합니다.

W&B run 의 완료를 표시하고 모든 데이터가 서버에 동기화되었는지 확인합니다. run 의 최종 상태는 종료 조건과 동기화 상태에 따라 결정됩니다.

Run 상태:
- Running: 데이터 로깅 중이거나 하트비트를 보내는 활성 run 입니다.
- Crashed: 예기치 않게 하트비트 전송이 중단된 run 입니다.
- Finished: 모든 데이터가 동기화된 채 정상적으로 완료된 run 입니다 (`exit_code=0`).
- Failed: 오류와 함께 종료된 run 입니다 (`exit_code!=0`).


**ARG:**
 
 - `exit_code`:  run 의 종료 상태를 나타내는 정수 값입니다. 0이면 성공, 그 외의 값은 run 이 실패한 것으로 간주합니다.
 - `quiet`:  더 이상 사용되지 않습니다. 로깅 상세 정도는 `wandb.Settings(quiet=...)`로 설정하세요.
