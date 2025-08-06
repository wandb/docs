---
title: 레지스트리 삭제
menu:
  default:
    identifier: ko-guides-core-registry-delete_registry
    parent: registry
weight: 8
---

이 페이지에서는 Team admin 또는 Registry admin 이 커스텀 registry 를 삭제하는 방법을 안내합니다. [core registry]({{< relref path="/guides/core/registry/registry_types#core-registry" lang="ko" >}})는 삭제할 수 없습니다.

- Team admin 은 조직 내의 어떤 커스텀 registry 도 삭제할 수 있습니다.
- Registry admin 은 자신이 생성한 커스텀 registry 를 삭제할 수 있습니다.

registry 를 삭제하면 해당 registry 에 속한 컬렉션들도 함께 삭제되지만, registry 와 연결된 artifact 는 삭제되지 않습니다. 해당 artifact 는 artifact 가 로그된 원래 프로젝트 에 남아 있게 됩니다.


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

`wandb` API의 `delete()` 메소드를 사용하여 프로그램적으로 registry 를 삭제할 수 있습니다. 다음 예시는 다음 과정을 보여줍니다.

1. 삭제하고자 하는 registry 를 `api.registry()`로 가져옵니다.
1. 반환된 registry 오브젝트 에서 `delete()` 메소드를 호출하여 registry 를 삭제합니다.

```python
import wandb

# W&B API 초기화
api = wandb.Api()

# 삭제하고자 하는 registry 가져오기
fetched_registry = api.registry("<registry_name>")

# registry 삭제
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. https://wandb.ai/registry/ 에 있는 **Registry** 앱으로 이동합니다.
2. 삭제하고자 하는 커스텀 registry 를 선택합니다.
3. 오른쪽 상단의 기어 아이콘을 클릭하여 registry 설정을 엽니다.
4. registry 를 삭제하려면, 설정 페이지 오른쪽 상단의 휴지통 아이콘을 클릭합니다.
5. 나타나는 모달 창에 삭제할 registry 의 이름을 입력하여 삭제 의사를 확인한 다음, **Delete** 를 클릭합니다.

{{% /tab %}}
{{< /tabpane >}}