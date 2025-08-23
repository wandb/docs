---
title: teardown()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-teardown
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




### <kbd>function</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

W&B 작업이 모두 완료되고 리소스가 해제될 때까지 대기합니다.

`run.finish()`로 명시적으로 종료되지 않은 모든 Run 을 완료하고, 모든 데이터가 업로드될 때까지 기다립니다.

`wandb.setup()`을 사용한 세션의 마지막에 이 함수를 호출하는 것을 권장합니다. 이 함수는 `atexit` 훅 내에서 자동으로 호출되지만, Python의 `multiprocessing` 모듈을 사용하는 환경 등 일부 설정에서는 신뢰할 수 없습니다.
