---
title: 각 Artifacts 버전은 얼마나 많은 스토리지를 사용하나요?
menu:
  support:
    identifier: ko-support-kb-articles-artifact_storage_version
support:
- Artifacts
- 스토리지
toc_hide: true
type: docs
url: /support/:filename
---

두 아티팩트 버전 사이에서 변경된 파일에 대해서만 저장 비용이 부과됩니다.

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifact deduplication" >}}

예를 들어, `animals`라는 이미지 아티팩트에 두 개의 이미지 파일 `cat.png`와 `dog.png`가 포함되어 있다고 가정해봅시다:

```
images
|-- cat.png (2MB) # `v0`에서 추가됨
|-- dog.png (1MB) # `v0`에서 추가됨
```

이 아티팩트는 `v0`라는 버전을 갖게 됩니다.

새로운 이미지인 `rat.png`를 추가하면, 다음과 같은 내용으로 새로운 아티팩트 버전 `v1`이 생성됩니다:

```
images
|-- cat.png (2MB) # `v0`에서 추가됨
|-- dog.png (1MB) # `v0`에서 추가됨
|-- rat.png (3MB) # `v1`에서 추가됨
```

버전 `v1`은 총 6MB의 파일을 추적하지만, `v0`와 3MB의 파일을 공유하고 있으므로 실제로는 3MB의 공간만 추가로 사용됩니다. 만약 `v1`을 삭제하면 `rat.png`와 관련된 3MB의 저장 공간이 반환됩니다. 반대로 `v0`를 삭제하면 `cat.png`와 `dog.png`의 저장 비용이 `v1`로 이전되어, `v1`의 저장 공간 사용량이 6MB로 증가합니다.