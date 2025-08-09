---
title: 'restore()

  '
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-restore
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>function</kbd> `restore`

```python
restore(
    name: 'str',
    run_path: 'str | None' = None,
    replace: 'bool' = False,
    root: 'str | None' = None
) → None | TextIO
```

클라우드 저장소에서 지정한 파일을 다운로드합니다.

파일은 현재 디렉토리 또는 run 디렉토리에 저장됩니다. 기본적으로 파일이 이미 존재하지 않는 경우에만 다운로드합니다.



**인자(Args):**
 
 - `name`:  파일 이름입니다.
 - `run_path`:  파일을 가져올 run 의 경로입니다. 예: `username/project_name/run_id`  wandb.init 이 호출되지 않았다면 필수입니다.
 - `replace`:  로컬에 이미 파일이 있어도 다시 다운로드할지 여부입니다.
 - `root`:  파일을 다운로드할 디렉토리입니다. 기본값은 현재 디렉토리 또는 wandb.init 이 호출된 경우 run 디렉토리입니다.



**반환(Returns):**  
파일을 찾지 못하면 None 을 반환하며, 그렇지 않으면 읽기 모드로 열린 파일 오브젝트를 반환합니다.



**예외(Raises):**
 
 - `CommError`:  W&B 가 W&B 백엔드에 연결할 수 없는 경우
 - `ValueError`:  파일을 찾을 수 없거나 run_path 를 찾을 수 없는 경우
```