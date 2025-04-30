---
title: Manage artifact storage and memory allocation
description: W&B Artifacts 의 스토리지, 메모리 할당을 관리합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-storage
    parent: manage-data
---

W&B는 기본적으로 미국에 위치한 개인 Google Cloud Storage 버킷에 아티팩트 파일을 저장합니다. 모든 파일은 저장 및 전송 중에 암호화됩니다.

민감한 파일의 경우, [Private Hosting]({{< relref path="/guides/hosting/" lang="ko" >}})을 설정하거나 [reference artifacts]({{< relref path="../track-external-files.md" lang="ko" >}})를 사용하는 것이 좋습니다.

트레이닝 중에 W&B는 로그, 아티팩트 및 설정 파일을 다음 로컬 디렉토리에 로컬로 저장합니다.

| 파일 | 기본 위치 | 기본 위치를 변경하려면 다음을 설정하십시오: |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `wandb.init`의 `dir` 또는 `WANDB_DIR` 환경 변수를 설정하십시오. |
| artifacts | `~/.cache/wandb` | `WANDB_CACHE_DIR` 환경 변수 |
| configs | `~/.config/wandb` | `WANDB_CONFIG_DIR` 환경 변수 |
| 업로드를 위한 Staging artifacts | `~/.cache/wandb-data/` | `WANDB_DATA_DIR` 환경 변수 |
| 다운로드된 artifacts | `./artifacts` | `WANDB_ARTIFACT_DIR` 환경 변수 |

환경 변수를 사용하여 W&B를 구성하는 방법에 대한 전체 가이드는 [환경 변수 참조]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 참조하십시오.

{{% alert color="secondary" %}}
`wandb`가 초기화된 머신에 따라 이러한 기본 폴더가 파일 시스템의 쓰기 가능한 부분에 위치하지 않을 수 있습니다. 이로 인해 오류가 발생할 수 있습니다.
{{% /alert %}}

### 로컬 아티팩트 캐시 정리

W&B는 공통 파일을 공유하는 버전 간의 다운로드 속도를 높이기 위해 아티팩트 파일을 캐시합니다. 시간이 지남에 따라 이 캐시 디렉토리가 커질 수 있습니다. [`wandb artifact cache cleanup`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-cache/" lang="ko" >}}) 명령을 실행하여 캐시를 정리하고 최근에 사용되지 않은 파일을 제거하십시오.

다음 코드 조각은 캐시 크기를 1GB로 제한하는 방법을 보여줍니다. 코드 조각을 복사하여 터미널에 붙여넣으십시오:

```bash
$ wandb artifact cache cleanup 1GB
```
