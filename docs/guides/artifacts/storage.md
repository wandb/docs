---
title: Manage artifact storage and memory allocation
description: W&B Artifacts의 저장소, 메모리 할당 관리.
displayed_sidebar: default
---

W&B는 기본적으로 미국에 위치한 비공개 Google 클라우드 Storage 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장 중일 때와 전송 중일 때 모두 암호화됩니다.

민감한 파일을 다룰 경우, [Private Hosting](../hosting/intro.md) 설정을 하거나 [reference artifacts](./track-external-files.md)를 사용하는 것을 권장합니다.

트레이닝 중에는 W&B가 아래의 로컬 디렉토리에 로그, 아티팩트, 그리고 설정 파일을 로컬로 저장합니다:

| 파일       | 기본 위치            | 기본 위치를 변경하려면 설정:                                            |
| --------- | ----------------- | ----------------------------------------------------------------- |
| logs      | `./wandb`         | `wandb.init` 내의 `dir` 또는 `WANDB_DIR` 환경 변수를 설정            |
| artifacts | `~/.cache/wandb`  | `WANDB_CACHE_DIR` 환경 변수를 설정                                     |
| configs   | `~/.config/wandb` | `WANDB_CONFIG_DIR` 환경 변수를 설정                                     |

:::caution
`wandb`가 초기화된 머신에 따라, 이 기본 폴더들이 파일 시스템의 쓰기 가능 부분에 위치하지 않을 수 있습니다. 이 경우 오류가 발생할 수 있습니다.
:::

### 로컬 아티팩트 캐시 정리

W&B는 파일을 공유하는 버전들 간의 다운로드 속도를 높이기 위해 아티팩트 파일을 캐시합니다. 시간이 지남에 따라 이 캐시 디렉토리가 커질 수 있습니다. 최근에 사용되지 않은 파일을 정리하고 캐시를 정리하려면 [`wandb artifact cache cleanup`](../../ref/cli/wandb-artifact/wandb-artifact-cache/README.md) 코맨드를 실행하세요.

다음 코드조각은 캐시 크기를 1GB로 제한하는 방법을 보여줍니다. 터미널에 코드조각을 복사하여 붙여넣으세요:

```bash
$ wandb artifact cache cleanup 1GB
```

### 각 아티팩트 버전이 얼마나 많은 저장 용량을 사용할까요?

두 아티팩트 버전 사이에서 변경된 파일만이 저장 비용을 발생시킵니다.

![아티팩트 "dataset"의 v1은 2/5 이미지가 다르기 때문에, 총 공간의 40%를 사용합니다.](/images/artifacts/artifacts-dedupe.PNG)

예를 들어, `animals`라는 이미지 아티팩트를 생성하고 cat.png와 dog.png라는 두 이미지 파일을 포함한다고 가정해봅시다:

```
images
|-- cat.png (2MB) # `v0`에 추가됨
|-- dog.png (1MB) # `v0`에 추가됨
```

이 아티팩트는 자동으로 버전 `v0`이 지정됩니다.

새로운 이미지 `rat.png`를 아티팩트에 추가하면 새로운 아티팩트 버전인 `v1`이 생성되며 다음의 내용을 갖습니다:

```
images
|-- cat.png (2MB) # `v0`에 추가됨
|-- dog.png (1MB) # `v0`에 추가됨
|-- rat.png (3MB) # `v1`에 추가됨
```

`v1`은 총 6MB의 파일을 추적하지만, `v0`와 공유하는 3MB 때문에 실제로는 3MB의 공간만 차지합니다. 만약 `v1`을 삭제하면 `rat.png`와 관련된 3MB의 저장 공간을 회수하게 됩니다. 만약 `v0`을 삭제하면, `v1`이 `cat.png`와 `dog.png`의 저장 비용을 인계받아 총 저장 용량이 6MB로 증가하게 됩니다.