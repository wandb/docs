---
title: 자동화
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-automation
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/automations.py >}}



저장된 W&B automation의 로컬 인스턴스입니다.

속성:
- action (Union): 이 automation이 트리거될 때 실행될 액션입니다.
- description (Optional): 이 automation에 대한 선택적 설명입니다.
- enabled (bool): 이 automation이 활성화되어 있는지 여부입니다. 활성화된 automation만 트리거됩니다.
- event (SavedEvent): 이 automation을 트리거하는 이벤트입니다.
- name (str): 이 automation의 이름입니다.
- scope (Union): 트리거링 이벤트가 발생해야 하는 범위입니다.
