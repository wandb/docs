---
title: 아티팩트 저장소 및 메모리 할당 관리
description: W&B Artifacts의 스토리지와 메모리 할당을 관리하세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B는 기본적으로 미국에 위치한 Google Cloud Storage 비공개 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장 시와 전송 시 모두 암호화됩니다.

민감한 파일의 경우, [Private Hosting]({{< relref path="/guides/hosting/" lang="ko" >}})을 설정하거나 [reference artifacts]({{< relref path="../track-external-files.md" lang="ko" >}}) 사용을 권장합니다.

트레이닝 중에는, W&B가 로그, Artifacts, 그리고 설정 파일을 아래의 로컬 디렉토리에 저장합니다:

| 파일 | 기본 위치 | 기본 위치를 변경하려면: |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `wandb.init`의 `dir` 값 또는 `WANDB_DIR` 환경 변수 설정 |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 환경 변수 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 환경 변수 |
| 업로드를 위한 staging artifacts  | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 환경 변수 |
| 다운로드된 artifacts | `./artifacts` | `WANDB_ARTIFACT_DIR` 환경 변수 |

환경 변수로 W&B 동작을 설정하는 방법에 대한 전체 가이드는 [환경 변수 레퍼런스]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 참고하세요.

{{% alert color="secondary" %}}
`wandb`가 초기화되는 머신에 따라 기본 폴더들이 파일 시스템에서 쓰기 가능한 위치에 없을 수 있습니다. 이런 경우 오류가 발생할 수 있습니다.
{{% /alert %}}

### 로컬 artifact 캐시 정리하기

W&B는 버전 간에 공유되는 파일의 다운로드 속도를 높이기 위해 artifact 파일을 캐시합니다. 시간이 지나면 이 캐시 디렉토리가 매우 커질 수 있습니다. 캐시를 정리하고 최근에 사용되지 않은 파일을 제거하려면 [`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ko" >}}) 명령어를 실행하세요.

아래 코드조각은 캐시의 최대 크기를 1GB로 제한하는 예시입니다. 터미널에 복사해서 사용하세요:

```bash
$ wandb artifact cache cleanup 1GB
```