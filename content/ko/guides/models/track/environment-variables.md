---
title: 환경 변수
description: W&B 환경 변수를 설정하세요.
menu:
  default:
    identifier: ko-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

자동화된 환경에서 스크립트를 실행할 때, 스크립트 실행 전에 환경 변수로 W&B 를 제어하거나 스크립트 내부에서 설정할 수 있습니다.

```bash
# 이 정보는 비밀이며 버전 관리에 체크인하지 않아야 합니다
WANDB_API_KEY=$YOUR_API_KEY
# 이름과 노트는 선택사항입니다
WANDB_NAME="My first run"
WANDB_NOTES="작은 러닝 레이트, 더 많은 정규화."
```

```bash
# wandb/settings 파일을 체크인하지 않는 경우에만 필요합니다
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# 스크립트가 cloud 와 동기화되지 않게 하려면
os.environ["WANDB_MODE"] = "offline"

# Run 객체 및 관련 클래스에 sweep ID 트래킹을 추가합니다
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## 선택적 환경 변수

원격 머신에서 인증을 설정하는 등 다양한 목적으로 아래 선택적 환경 변수를 사용할 수 있습니다.

| 변수 이름 | 용도 |
| --------------------------- | ---------- |
| `WANDB_ANONYMOUS` | 사용자가 비공개 URL 로 익명 run 을 생성할 수 있도록 `allow`, `never`, `must` 중 하나로 설정합니다. |
| `WANDB_API_KEY` | 계정에 연결된 인증 키를 설정합니다. 자신의 키는 [설정 페이지](https://app.wandb.ai/settings)에서 확인할 수 있습니다. 원격 머신에서 `wandb login`을 실행하지 않았다면 반드시 이 값을 설정해야 합니다. |
| `WANDB_BASE_URL` | [wandb/local]({{< relref path="/guides/hosting/" lang="ko" >}})을 사용하는 경우 이 환경 변수를 `http://YOUR_IP:YOUR_PORT` 값으로 설정해야 합니다. |
| `WANDB_CACHE_DIR` | 기본값은 \~/.cache/wandb 입니다. 다른 위치를 원할 경우 이 환경 변수로 변경할 수 있습니다. |
| `WANDB_CONFIG_DIR` | 기본값은 \~/.config/wandb 입니다. 다른 위치를 원할 경우 이 환경 변수로 변경할 수 있습니다. |
| `WANDB_CONFIG_PATHS` | wandb.config 에 로드할 yaml 파일 목록을 콤마로 구분해 나열합니다. 자세한 내용은 [config]({{< relref path="./config.md#file-based-configs" lang="ko" >}})를 참고하세요. |
| `WANDB_CONSOLE` | "off"로 설정하면 stdout / stderr 로그가 비활성화됩니다. 지원되는 환경에서는 기본값이 "on"입니다. |
| `WANDB_DATA_DIR` | staging artifact를 업로드할 위치입니다. 기본 경로는 사용중인 플랫폼에 따라 다르며, `platformdirs` Python 패키지의 `user_data_dir` 값을 사용합니다. 해당 디렉토리가 존재하고, 실행하는 사용자가 쓰기 권한이 있는지 확인하세요. |
| `WANDB_DIR` | 생성되는 모든 파일을 저장할 위치입니다. 설정하지 않으면 트레이닝 스크립트 기준 상대경로에 wandb 디렉토리가 기본값입니다. 디렉토리가 존재하고, 실행 사용자가 쓰기 권한이 있는지 확인하세요. 다운로드된 artifact 위치는 이 변수로 지정되지 않으며, `WANDB_ARTIFACT_DIR` 환경 변수로 별도로 설정할 수 있습니다. |
| `WANDB_ARTIFACT_DIR` | 다운로드된 모든 artifact 를 저장할 위치입니다. 설정하지 않으면 트레이닝 스크립트 기준 상대경로에 artifacts 디렉토리가 기본값입니다. 디렉토리가 존재하고, 실행 사용자가 쓰기 권한이 있는지 확인하세요. 생성되는 메타데이터 파일 위치는 이 변수로 지정되지 않으며, `WANDB_DIR` 환경 변수로 별도로 설정할 수 있습니다. |
| `WANDB_DISABLE_GIT` | W&B 가 git 저장소를 탐색하고 최근 커밋/차이(diff)를 기록하는 것을 막습니다. |
| `WANDB_DISABLE_CODE` | true로 설정 시 W&B 가 노트북이나 git diff 를 저장하지 않습니다. 단, git 저장소 내에서는 현재 커밋 정보는 계속 저장됩니다. |
| `WANDB_DOCKER` | docker 이미지 다이제스트를 설정해 run 복원을 활성화합니다. wandb docker 명령어로 자동 설정됩니다. 이미지 다이제스트는 `wandb docker my/image/name:tag --digest` 명령어로 얻을 수 있습니다. |
| `WANDB_ENTITY` | run 에 연결된 entity 를 지정합니다. 트레이닝 스크립트 디렉토리에서 `wandb init`을 실행했다면 _wandb_ 라는 디렉토리를 만들고, 기본 entity 값이 저장됩니다(소스 관리에 체크인 가능). 해당 파일 생성을 원하지 않거나 값을 덮어쓰기 원한다면 환경 변수를 사용하세요. |
| `WANDB_ERROR_REPORTING` | false 로 설정 시 W&B 가 치명적 오류를 자체 추적 시스템에 로깅하지 않습니다. |
| `WANDB_HOST` | 시스템에서 제공하는 호스트명이 아닌, wandb 인터페이스에 표시하고 싶은 호스트명을 직접 지정할 수 있습니다. |
| `WANDB_IGNORE_GLOBS` | 무시할 파일 glob 패턴을 콤마로 구분해 나열합니다. 이 파일들은 cloud에 동기화되지 않습니다. |
| `WANDB_JOB_NAME` | W&B 가 생성하는 job 의 이름을 지정합니다. |
| `WANDB_JOB_TYPE` | "training" 또는 "evaluation" 등 다양한 run 타입을 지정합니다. 자세한 내용은 [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})에서 확인하세요. |
| `WANDB_MODE` | "offline"으로 설정하면 run 메타데이터를 로컬에만 저장하고 cloud 와 동기화하지 않습니다. `disabled`로 설정 시 W&B 를 완전히 끌 수 있습니다. |
| `WANDB_NAME` | run의 사람이 읽을 수 있는 이름입니다. 설정하지 않으면 W&B 가 자동으로 무작위로 정해줍니다. |
| `WANDB_NOTEBOOK_NAME` | jupyter 환경에서 실행 중이라면, 이 변수로 노트북의 이름을 설정할 수 있습니다. 자동으로 감지하려고 시도합니다. |
| `WANDB_NOTES` | run에 대한 긴 설명을 남길 수 있습니다. 마크다운이 허용되며, 이후 UI에서도 수정 가능합니다. |
| `WANDB_PROJECT` | run 에 연결된 project 입니다. `wandb init` 명령어로도 설정할 수 있지만, 환경 변수가 있을 경우 이를 우선 사용합니다. |
| `WANDB_RESUME` | 기본값은 _never_ 입니다. _auto_로 설정하면 실패한 run 을 자동으로 이어서 실행합니다. _must_로 설정하면 run 이 반드시 존재해야 시작됩니다. 고유 id 를 항상 직접 생성하고 싶다면 _allow_로 설정하고 `WANDB_RUN_ID`를 항상 지정하세요. |
| `WANDB_RUN_GROUP` | 실험 이름을 지정해 여러 run 을 자동으로 그룹화할 수 있습니다. 자세한 내용은 [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})에서 확인하세요. |
| `WANDB_RUN_ID` | 스크립트 실행마다 전역적으로 고유한 문자열(프로젝트 단위)을 지정합니다. 64자 이하만 허용되며, 영숫자가 아닌 문자는 대시(-)로 변환됩니다. 장애가 발생할 경우 기존 run을 재개하는 데 사용할 수 있습니다. |
| `WANDB_QUIET` | `true`로 설정 시, 표준 출력에 기록되는 로그가 최소화되어 심각한 정보만 표시됩니다. 이 변수가 설정될 경우 모든 로그는 `$WANDB_DIR/debug.log`에 저장됩니다. |
| `WANDB_SILENT` | `true`로 설정하면 W&B 로그 메시지가 완전히 표시되지 않습니다. 스크립트 코맨드 실행에 유용합니다. 이 변수가 설정될 경우 모든 로그는 `$WANDB_DIR/debug.log`에 저장됩니다. |
| `WANDB_SHOW_RUN` | 운영체제가 지원하는 경우, run URL 이 자동으로 브라우저에서 열리도록 하려면 `true`로 설정하세요. |
| `WANDB_SWEEP_ID` | `Run` 객체와 관련 클래스에 sweep ID 트래킹을 추가하고, UI에 표시합니다. |
| `WANDB_TAGS` | run 에 적용할 태그들을 콤마로 구분해 나열합니다. |
| `WANDB_USERNAME` | run 에 연결할 팀 멤버의 사용자 이름입니다. 서비스 계정 API 키와 함께 사용해 자동화 run 의 소유권을 개별 멤버로 지정할 수 있습니다. |
| `WANDB_USER_EMAIL` | run 에 연결할 팀 멤버의 이메일입니다. 서비스 계정 API 키와 함께 사용해 자동화 run 의 소유권을 개별 멤버로 지정할 수 있습니다. |

## Singularity 환경 사용 시

[Singularity](https://singularity.lbl.gov/index.html) 컨테이너에서 실행하는 경우 위 환경 변수 앞에 `SINGULARITYENV_`를 붙이면 사용할 수 있습니다. Singularity 환경 변수에 대한 자세한 내용은 [여기](https://singularity.lbl.gov/docs-environment-metadata#environment)를 참고하세요.

## AWS 에서 실행하기

AWS에서 배치 작업을 실행한다면, W&B 자격 증명을 이용해 인증을 손쉽게 할 수 있습니다. [설정 페이지](https://app.wandb.ai/settings)에서 자신의 API 키를 확인한 뒤, [AWS batch job spec](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)에 `WANDB_API_KEY` 환경 변수를 추가하세요.