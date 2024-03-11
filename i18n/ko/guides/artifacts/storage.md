---
description: Manage storage, memory allocation of W&B Artifacts.
displayed_sidebar: default
---

# 저장소

<head>
    <title>아티팩트 저장소</title>
</head>

W&B는 기본적으로 미국에 위치한 비공개 Google 클라우드 저장소 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장될 때와 전송될 때 암호화됩니다.

민감한 파일의 경우, [Private Hosting](../hosting/intro.md)을 설정하거나 [참조 아티팩트](./track-external-files.md)를 사용하는 것이 좋습니다.

트레이닝 중, W&B는 로그, 아티팩트, 설정 파일을 다음 로컬 디렉토리에 저장합니다:

| 파일      | 기본 위치          | 기본 위치 변경 설정:                                              |
| --------- | ----------------- | ----------------------------------------------------------------- |
| 로그      | `./wandb`         | `wandb.init`의 `dir` 또는 `WANDB_DIR` 환경 변수 설정              |
| 아티팩트 | `~/.cache/wandb`  | `WANDB_CACHE_DIR` 환경 변수 설정                                   |
| 설정      | `~/.config/wandb` | `WANDB_CONFIG_DIR` 환경 변수 설정                                  |


:::caution
`wandb`가 초기화된 기계에 따라, 이 기본 폴더들은 파일 시스템의 쓸 수 있는 부분에 위치하지 않을 수 있습니다. 이는 오류를 유발할 수 있습니다.
:::

### 로컬 아티팩트 캐시 정리

W&B는 버전 간에 공통 파일을 공유하는 다운로드 속도를 높이기 위해 아티팩트 파일을 캐시합니다. 시간이 지남에 따라 이 캐시 디렉토리는 커질 수 있습니다. 캐시를 정리하고 최근에 사용되지 않은 파일을 제거하려면 [`wandb artifact cache cleanup`](../../ref/cli/wandb-artifact/wandb-artifact-cache/README.md) 코맨드를 실행하세요.

다음 코드조각은 캐시의 크기를 1GB로 제한하는 방법을 보여줍니다. 코드조각을 복사하여 터미널에 붙여넣으세요:

```bash
$ wandb artifact cache cleanup 1GB
```

### 각 아티팩트 버전이 사용하는 저장소 용량은 얼마나 됩니까?

두 아티팩트 버전 간에 변경된 파일만 저장 비용을 발생시킵니다.

![아티팩트 "dataset"의 v1은 2/5 이미지만 다르므로, 공간의 40%만을 사용합니다.](@site/static/images/artifacts/artifacts-dedupe.PNG)

예를 들어, `animals`이라는 이미지 아티팩트를 만들고, 이 아티팩트에 cat.png와 dog.png 두 이미지 파일을 포함시켰다고 가정해봅시다:

```
images
|-- cat.png (2MB) # `v0`에 추가됨
|-- dog.png (1MB) # `v0`에 추가됨
```

이 아티팩트는 자동으로 `v0`이라는 버전이 할당됩니다.

새 이미지 `rat.png`를 아티팩트에 추가하면, 새로운 아티팩트 버전 `v1`이 생성되고, 다음과 같은 내용을 가지게 됩니다:

```
images
|-- cat.png (2MB) # `v0`에 추가됨
|-- dog.png (1MB) # `v0`에 추가됨
|-- rat.png (3MB) # `v1`에 추가됨
```

`v1`은 총 6MB의 파일을 추적하지만, `v0`과 공통으로 공유하는 나머지 3MB 때문에 실제로는 3MB의 공간만 차지합니다. `v1`을 삭제하면, `rat.png`와 연관된 3MB의 저장소를 회수합니다. 만약 `v0`을 삭제한다면, `v1`은 `cat.png`와 `dog.png`의 저장 비용을 상속받아 저장 크기가 6MB가 됩니다.