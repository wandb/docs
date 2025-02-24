---
title: How much storage does each artifact version use?
menu:
  support:
    identifier: ko-support-artifact_storage_version
tags:
- artifacts
- storage
toc_hide: true
type: docs
---

두 아티팩트 버전 간에 변경된 파일만 스토리지 비용이 발생합니다.

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="아티팩트 'dataset'의 v1 버전은 5개의 이미지 중 2개만 다르므로 공간의 40%만 차지합니다." >}}

`animals`라는 이미지 아티팩트가 있고, 이 아티팩트가 두 개의 이미지 파일인 `cat.png`와 `dog.png`를 포함한다고 가정해 보겠습니다.

```
images
|-- cat.png (2MB) # `v0`에서 추가됨
|-- dog.png (1MB) # `v0`에서 추가됨
```

이 아티팩트는 버전 `v0`을 받습니다.

새 이미지 `rat.png`를 추가하면 다음 콘텐츠와 함께 새 아티팩트 버전 `v1`이 생성됩니다.

```
images
|-- cat.png (2MB) # `v0`에서 추가됨
|-- dog.png (1MB) # `v0`에서 추가됨
|-- rat.png (3MB) # `v1`에서 추가됨
```

버전 `v1`은 총 6MB를 추적하지만 나머지 3MB를 `v0`과 공유하므로 3MB의 공간만 차지합니다. `v1`을 삭제하면 `rat.png`와 연결된 3MB의 스토리지가 회수됩니다. `v0`을 삭제하면 `cat.png` 및 `dog.png`의 스토리지 비용이 `v1`로 전송되어 스토리지 크기가 6MB로 늘어납니다.
