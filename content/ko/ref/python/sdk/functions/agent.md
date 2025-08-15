---
title: agent()
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-agent
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb_agent.py >}}




### <kbd>function</kbd> `agent`

```python
agent(
    sweep_id: str,
    function: Optional[Callable] = None,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    count: Optional[int] = None
) → None
```

하나 이상의 스윕 에이전트를 시작합니다.

스윕 에이전트는 `sweep_id`를 사용하여 자신이 속한 스윕, 실행할 함수, 그리고(선택적으로) 몇 개의 에이전트를 실행할지 결정합니다.



**ARG:**
 
 - `sweep_id`:  스윕의 고유 식별자입니다. 스윕 ID는 W&B CLI 또는 Python SDK에서 생성됩니다.
 - `function`:  스윕 설정에 지정된 "program" 대신 호출할 함수입니다.
 - `entity`:  스윕에서 생성된 W&B run을 전송하려는 사용자명 또는 팀명입니다. 지정한 entity가 이미 존재해야 합니다. 만약 entity를 명시하지 않으면 기본 entity(보통 본인의 사용자명)로 run이 전송됩니다.
 - `project`:  스윕에서 생성된 W&B run이 전송될 Project의 이름입니다. project를 지정하지 않으면 run은 "Uncategorized"라는 프로젝트로 전송됩니다.
 - `count`:  시도할 스윕 설정 트라이얼의 개수입니다.
