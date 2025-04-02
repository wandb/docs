---
title: Environment variables
description: W&B 환경 변수를 설정하세요.
menu:
  default:
    identifier: ko-guides-models-track-environment-variables
    parent: experiments
weight: 9
---

자동화된 환경에서 스크립트를 실행할 때 스크립트 실행 전 또는 스크립트 내에서 설정된 환경 변수로 **wandb**를 제어할 수 있습니다.

```bash
# 이것은 비밀이며 버전 관리 시스템에 체크인되어서는 안 됩니다.
WANDB_API_KEY=$YOUR_API_KEY
# 이름과 노트는 선택 사항입니다.
WANDB_NAME="나의 첫 번째 run"
WANDB_NOTES="더 작은 학습률, 더 많은 정규화."
```

```bash
# wandb/settings 파일을 체크인하지 않은 경우에만 필요합니다.
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# 스크립트가 클라우드에 동기화되는 것을 원하지 않는 경우
os.environ["WANDB_MODE"] = "offline"

# 스윕 ID 추적을 Run 오브젝트 및 관련 클래스에 추가
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## 선택적 환경 변수

이러한 선택적 환경 변수를 사용하여 원격 머신에서 인증을 설정하는 등의 작업을 수행합니다.

| 변수 이름 | 사용법 |
| --------------------------- | ---------- |
| **WANDB_ANONYMOUS** | 사용자가 비밀 URL로 익명 run을 생성하도록 허용하려면 이 변수를 `allow`, `never` 또는 `must`로 설정합니다. |
| **WANDB_API_KEY** | 계정과 연결된 인증 키를 설정합니다. 키는 [설정 페이지](https://app.wandb.ai/settings)에서 찾을 수 있습니다. 원격 머신에서 `wandb login`이 실행되지 않은 경우 이 변수를 설정해야 합니다. |
| **WANDB_BASE_URL** | [wandb/local]({{< relref path="/guides/hosting/" lang="ko" >}})을 사용하는 경우 이 환경 변수를 `http://YOUR_IP:YOUR_PORT`로 설정해야 합니다. |
| **WANDB_CACHE_DIR** | 기본값은 \~/.cache/wandb이며, 이 환경 변수로 이 위치를 재정의할 수 있습니다. |
| **WANDB_CONFIG_DIR** | 기본값은 \~/.config/wandb이며, 이 환경 변수로 이 위치를 재정의할 수 있습니다. |
| **WANDB_CONFIG_PATHS** | wandb.config에 로드할 쉼표로 구분된 yaml 파일 목록입니다. [config]({{< relref path="./config.md#file-based-configs" lang="ko" >}})를 참조하십시오. |
| **WANDB_CONSOLE** | stdout / stderr 로깅을 비활성화하려면 이 변수를 "off"로 설정합니다. 기본적으로 이를 지원하는 환경에서는 "on"으로 설정됩니다. |
| **WANDB_DATA_DIR** | 스테이징 Artifacts가 업로드되는 위치입니다. 기본 위치는 `platformdirs` Python 패키지의 `user_data_dir` 값을 사용하기 때문에 플랫폼에 따라 다릅니다. |
| **WANDB_DIR** | 트레이닝 스크립트를 기준으로 _wandb_ 디렉토리가 아닌 여기에 생성된 모든 파일을 저장하려면 이 변수를 절대 경로로 설정합니다. _이 디렉토리가 존재하고 프로세스가 실행되는 사용자가 쓸 수 있는지 확인하십시오_. 이는 다운로드된 Artifacts의 위치에는 영향을 미치지 않으며, 대신 _WANDB_ARTIFACT_DIR_을 사용하여 설정할 수 있습니다. |
| **WANDB_ARTIFACT_DIR** | 트레이닝 스크립트를 기준으로 _artifacts_ 디렉토리가 아닌 여기에 다운로드된 모든 Artifacts를 저장하려면 이 변수를 절대 경로로 설정합니다. 이 디렉토리가 존재하고 프로세스가 실행되는 사용자가 쓸 수 있는지 확인하십시오. 이는 생성된 메타데이터 파일의 위치에는 영향을 미치지 않으며, 대신 _WANDB_DIR_을 사용하여 설정할 수 있습니다. |
| **WANDB_DISABLE_GIT** | wandb가 git 저장소를 검색하고 최신 커밋/diff를 캡처하지 못하도록 합니다. |
| **WANDB_DISABLE_CODE** | wandb가 노트북 또는 git diff를 저장하지 못하도록 하려면 이 변수를 true로 설정합니다. git 저장소에 있는 경우 현재 커밋은 계속 저장됩니다. |
| **WANDB_DOCKER** | run 복원을 활성화하려면 이 변수를 docker 이미지 다이제스트로 설정합니다. 이는 wandb docker 코맨드로 자동 설정됩니다. `wandb docker my/image/name:tag --digest`를 실행하여 이미지 다이제스트를 얻을 수 있습니다. |
| **WANDB_ENTITY** | run과 연결된 entity입니다. 트레이닝 스크립트의 디렉토리에서 `wandb init`를 실행한 경우 _wandb_라는 디렉토리가 생성되고 소스 제어에 체크인할 수 있는 기본 entity가 저장됩니다. 해당 파일을 생성하지 않거나 파일을 재정의하려는 경우 환경 변수를 사용할 수 있습니다. |
| **WANDB_ERROR_REPORTING** | wandb가 심각한 오류를 오류 추적 시스템에 로깅하지 못하도록 하려면 이 변수를 false로 설정합니다. |
| **WANDB_HOST** | 시스템에서 제공하는 호스트 이름을 사용하지 않으려는 경우 wandb 인터페이스에 표시할 호스트 이름으로 설정합니다. |
| **WANDB_IGNORE_GLOBS** | 무시할 파일 glob의 쉼표로 구분된 목록으로 설정합니다. 이러한 파일은 클라우드에 동기화되지 않습니다. |
| **WANDB_JOB_NAME** | `wandb`로 생성된 모든 jobs의 이름을 지정합니다. |
| **WANDB_JOB_TYPE** | run의 다양한 유형을 나타내기 위해 "training" 또는 "evaluation"과 같은 job 유형을 지정합니다. 자세한 내용은 [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})을 참조하십시오. |
| **WANDB_MODE** | 이 변수를 "offline"으로 설정하면 wandb가 run 메타데이터를 로컬에 저장하고 서버에 동기화하지 않습니다. 이 변수를 `disabled`로 설정하면 wandb가 완전히 꺼집니다. |
| **WANDB_NAME** | run의 사람이 읽을 수 있는 이름입니다. 설정하지 않으면 임의로 생성됩니다. |
| **WANDB_NOTEBOOK_NAME** | jupyter에서 실행 중인 경우 이 변수로 노트북 이름을 설정할 수 있습니다. 자동으로 감지하려고 시도합니다. |
| **WANDB_NOTES** | run에 대한 더 긴 메모입니다. Markdown이 허용되며 나중에 UI에서 편집할 수 있습니다. |
| **WANDB_PROJECT** | run과 연결된 project입니다. 이는 `wandb init`로도 설정할 수 있지만 환경 변수가 값을 재정의합니다. |
| **WANDB_RESUME** | 기본적으로 이는 _never_로 설정됩니다. _auto_로 설정하면 wandb가 실패한 run을 자동으로 재개합니다. _must_로 설정하면 시작 시 run이 강제로 존재합니다. 항상 고유한 ID를 생성하려면 _allow_로 설정하고 항상 **WANDB_RUN_ID**를 설정합니다. |
| **WANDB_RUN_GROUP** | run을 자동으로 그룹화할 실험 이름을 지정합니다. 자세한 내용은 [grouping]({{< relref path="/guides/models/track/runs/grouping.md" lang="ko" >}})을 참조하십시오. |
| **WANDB_RUN_ID** | 스크립트의 단일 run에 해당하는 전역적으로 고유한 문자열(project당)로 설정합니다. 64자 이하여야 합니다. 모든 단어가 아닌 문자는 대시로 변환됩니다. 이는 실패 시 기존 run을 재개하는 데 사용할 수 있습니다. |
| **WANDB_SILENT** | wandb 로그 문을 숨기려면 이 변수를 **true**로 설정합니다. 이 변수를 설정하면 모든 로그가 **WANDB_DIR**/debug.log에 기록됩니다. |
| **WANDB_SHOW_RUN** | 운영 체제에서 지원하는 경우 run URL로 브라우저를 자동으로 열려면 이 변수를 **true**로 설정합니다. |
| **WANDB_SWEEP_ID** | 스윕 ID 추적을 `Run` 오브젝트 및 관련 클래스에 추가하고 UI에 표시합니다. |
| **WANDB_TAGS** | run에 적용할 쉼표로 구분된 태그 목록입니다. |
| **WANDB_USERNAME** | run과 연결된 팀 구성원의 사용자 이름입니다. 이는 서비스 계정 API 키와 함께 사용하여 자동화된 run을 팀 구성원에게 귀속시키는 데 사용할 수 있습니다. |
| **WANDB_USER_EMAIL** | run과 연결된 팀 구성원의 이메일입니다. 이는 서비스 계정 API 키와 함께 사용하여 자동화된 run을 팀 구성원에게 귀속시키는 데 사용할 수 있습니다. |

## Singularity 환경

[Singularity](https://singularity.lbl.gov/index.html)에서 컨테이너를 실행하는 경우 위의 변수 앞에 **SINGULARITYENV_**를 붙여 환경 변수를 전달할 수 있습니다. Singularity 환경 변수에 대한 자세한 내용은 [여기](https://singularity.lbl.gov/docs-environment-metadata#environment)에서 확인할 수 있습니다.

## AWS에서 실행

AWS에서 배치 jobs를 실행하는 경우 W&B 자격 증명으로 머신을 쉽게 인증할 수 있습니다. [설정 페이지](https://app.wandb.ai/settings)에서 API 키를 가져오고 [AWS 배치 job 사양](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)에서 `WANDB_API_KEY` 환경 변수를 설정합니다.
