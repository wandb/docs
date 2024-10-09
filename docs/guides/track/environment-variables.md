---
title: Environment variables
description: W&B 환경 변수를 설정하세요.
displayed_sidebar: default
---

When you're running a script in an automated environment, you can control **wandb** with environment variables set before the script runs or within the script.

```bash
# 이 키는 비밀이며 버전 관리에 넣어선 안됩니다.
WANDB_API_KEY=$YOUR_API_KEY
# 이름과 노트는 선택 사항입니다.
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# wandb/settings 파일을 체크인하지 않는 경우에만 필요합니다.
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# 클라우드에 스크립트를 동기화하지 않으려면
os.environ["WANDB_MODE"] = "offline"
```

## 선택적인 환경 변수

원격 장치에서 인증을 설정하는 등의 작업을 하기 위해 이 선택적인 환경 변수를 사용하십시오.

| 변수 이름 | 사용법 |
| --------------------------- | ---------- |
| **WANDB_ANONYMOUS**        | 사용자가 비밀 URL로 익명 run을 만들 수 있도록 "allow", "never", "must" 중 하나로 설정합니다.                                                    |
| **WANDB_API_KEY**         | 계정과 연관된 인증 키를 설정합니다. 키는 [설정 페이지](https://app.wandb.ai/settings)에서 찾을 수 있습니다. 원격 장치에서 `wandb login`이 실행되지 않은 경우 이 값을 설정해야 합니다.               |
| **WANDB_BASE_URL**        | [wandb/local](../hosting/intro.md)을 사용하는 경우 이 환경 변수를 `http://YOUR_IP:YOUR_PORT`로 설정해야 합니다.       |
| **WANDB_CACHE_DIR**       | 기본 값은 \~/.cache/wandb이며, 이 환경 변수로 이 위치를 덮어쓸 수 있습니다.                  |
| **WANDB_CONFIG_DIR**      | 기본 값은 \~/.config/wandb이며, 이 환경 변수로 이 위치를 덮어쓸 수 있습니다.                             |
| **WANDB_CONFIG_PATHS**    | wandb.config에 로드할 yaml 파일의 쉼표로 구분된 목록입니다. [config](./config.md#file-based-configs)를 참조하세요.                                          |
| **WANDB_CONSOLE**          | "off"로 설정하면 stdout/stderr 로그 작성을 비활성화합니다. 지원되는 환경에서는 기본 값이 "on"입니다.                                          |
| **WANDB_DIR**              | 모든 생성된 파일을 해당 위치에 저장하도록 절대 경로로 설정합니다. _이 디렉토리가 존재하는지, 그리고 프로세스가 실행되는 사용자가 해당 위치에 쓸 수 있는지 확인하십시오._ |
| **WANDB_DISABLE_GIT**     | git 저장소를 검색하고 최신 커밋/차이를 캡처하는 것을 방지합니다.      |
| **WANDB_DISABLE_CODE**    | wandb가 노트북이나 git 차이를 저장하는 것을 방지하려면 true로 설정합니다. git 리포에 있는 경우 현재 커밋은 여전히 저장됩니다.                   |
| **WANDB_DOCKER**           | run을 복원할 수 있도록 도커 이미지 다이제스트를 설정합니다. 이 값은 자동으로 wandb docker 코맨드로 설정됩니다. `wandb docker my/image/name:tag --digest`를 실행하여 이미지 다이제스트를 얻을 수 있습니다.    |
| **WANDB_ENTITY**           | run과 관련된 엔티티입니다. 트레이닝 스크립트의 디렉토리에서 `wandb init`을 실행한 경우 _wandb_라는 디렉토리가 생성되고 기본 엔티티가 소스 제어에 체크인될 수 있습니다. 그 파일을 생성하고 싶지 않거나 파일을 덮어쓰고 싶다면 환경 변수를 사용할 수 있습니다. |
| **WANDB_ERROR_REPORTING** | wandb가 치명적인 오류를 그 자체의 오류 추적 시스템에 기록하지 못하도록 하려면 false로 설정하세요.                             |
| **WANDB_HOST**             | 시스템에서 제공하는 호스트 이름을 사용하고 싶지 않을 경우 이 값을 wandb 인터페이스에서에 표시할 호스트 이름으로 설정하세요.                                |
| **WANDB_IGNORE_GLOBS**    | 무시할 파일 글로브 목록을 쉼표로 구분해 설정하세요. 이러한 파일은 클라우드로 동기화되지 않습니다.                              |
| **WANDB_JOB_NAME**        | `wandb`가 생성한 job의 이름을 지정하세요. 추가 정보는 [create a job](../launch/create-launch-job.md)을 참조하세요.                                                                                                                                                                                                                        |
| **WANDB_JOB_TYPE**        | "training"이나 "evaluation"처럼 run의 다른 유형을 나타내기 위해 job 유형을 지정하세요. 자세한 내용은 [grouping](../runs/grouping.md)을 참조하세요.               |
| **WANDB_MODE**             | "offline"으로 설정하면 wandb가 run 메타데이터를 로컬에 저장하고 서버에 동기화하지 않습니다. "disabled"로 설정하면 wandb가 완전히 꺼집니다.                  |
| **WANDB_NAME**             | 사용자가 읽을 수 있는 run의 이름입니다. 설정하지 않을 경우 자동으로 생성됩니다.                       |
| **WANDB_NOTEBOOK_NAME**   | jupyter에서 실행 중인 경우 이 변수로 notebook의 이름을 설정할 수 있습니다. 자동으로 탐지하려고 노력합니다.                    |
| **WANDB_NOTES**            | 당신의 run에 대한 긴 노트입니다. 마크다운이 허용되며 나중에 UI에서 편집할 수 있습니다.                                    |
| **WANDB_PROJECT**          | run과 관련된 프로젝트입니다. `wandb init`으로도 설정할 수 있지만 환경 변수가 값을 무시합니다.                               |
| **WANDB_RESUME**           | 기본적으로 _never_로 설정됩니다. _auto_로 설정하면 wandb가 자동으로 실패한 run을 재개합니다. _must_로 설정하면 시작 시 run이 있어야 합니다. 항상 고유한 ID를 생성하고 싶다면 _allow_로 설정하고 항상 **WANDB_RUN_ID**를 설정하세요.      |
| **WANDB_RUN_GROUP**       | 실험 이름을 지정하여 run을 자동으로 그룹화합니다. 자세한 정보는 [grouping](../runs/grouping.md)을 참조하세요.                                 |
| **WANDB_RUN_ID**          | 스크립트의 단일 run에 해당하는 전역적으로 고유한 문자열(프로젝트당)로 설정합니다. 64자 이내여야 하며 모든 비단어 문자는 대시로 변환됩니다. 실패 시 기존 run을 다시 시작하는 용도로 사용할 수 있습니다.      |
| **WANDB_SILENT**           | wandb 로그 문구를 무음으로 하려면 **true**로 설정하세요. 이 경우 모든 로그는 **WANDB_DIR**/debug.log로 작성됩니다.               |
| **WANDB_SHOW_RUN**        | 운영 체제가 이를 지원하는 경우 자동으로 브라우저에서 run URL을 열도록 하려면 **true**로 설정하세요.        |
| **WANDB_TAGS**             | run에 적용할 태그의 쉼표로 구분된 목록입니다.                  |
| **WANDB_USERNAME**         | 팀의 멤버와 관련된 사용자의 이름입니다. 이 옵션은 서비스 계정 API 키와 함께 사용하여 자동 run을 팀 멤버에게 지정할 수 있습니다.               |
| **WANDB_USER_EMAIL**      | 팀의 멤버와 관련된 사용자의 이메일입니다. 이 옵션은 서비스 계정 API 키와 함께 사용하여 자동 run을 팀 멤버에게 지정할 수 있습니다.            |

## Singularity 환경

[Singularity](https://singularity.lbl.gov/index.html) 컨테이너를 실행하는 경우, 위의 변수들에 **SINGULARITYENV_**를 붙여서 환경 변수를 전달할 수 있습니다. Singularity 환경 변수에 대한 자세한 내용은 [여기](https://singularity.lbl.gov/docs-environment-metadata#environment)에서 확인할 수 있습니다.

## AWS에서 실행하기

AWS에서 배치 job을 실행하는 경우, W&B 자격 증명을 사용하여 기계를 인증하는 것은 쉽습니다. [설정 페이지](https://app.wandb.ai/settings)에서 API 키를 가져와, [AWS 배치 job 스펙](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)에 WANDB_API_KEY 환경 변수를 설정합니다.

## 자주 묻는 질문

### 자동 run 및 서비스 계정

자동 테스트나 W&B에 로그를 남기는 내부 툴을 포함하는 경우 팀 설정 페이지에서 **Service Account**를 생성하세요. 이를 통해 자동 job에 대해 서비스 API 키를 사용할 수 있습니다. 서비스 계정 job을 특정 사용자에게 지정하려면 **WANDB_USERNAME** 또는 **WANDB_USER_EMAIL** 환경 변수를 사용할 수 있습니다.

![팀 설정 페이지에서 자동 job을 위한 서비스 계정을 생성합니다.](/images/track/common_questions_automate_runs.png)

이는 연속 통합(continuous integration)과 자동화된 단위 테스트를 설정할 때 TravisCI나 CircleCI 같은 툴에 유용합니다.

### 환경 변수가 wandb.init()에 전달되는 파라미터를 덮어쓰나요?

`wandb.init`에 전달된 인수는 환경보다 우선합니다. 환경 변수가 설정되지 않은 경우 시스템 기본값이 아닌 기본값을 사용하려면 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`를 호출할 수 있습니다.

### 로그 끄기

코맨드 `wandb offline`은 환경 변수 `WANDB_MODE=offline`을 설정합니다. 이는 장치에서 원격 wandb 서버로의 데이터 동기화를 중단합니다. 여러 프로젝트가 있는 경우, 모든 로그된 데이터는 W&B 서버에 동기화되지 않습니다.

경고 메시지를 조용히 하기 위한 방법:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### 공유 장치에서 여러 wandb 사용자

공유 장치를 사용하는 경우 다른 사람이 wandb 사용자일 때, run이 항상 올바른 계정으로 로그되도록 하는 것은 쉽습니다. WANDB_API_KEY 환경 변수를 설정하여 인증하세요. 환경에서 이것을 소스하면 로그인할 때 올바른 자격 증명을 가지게 되거나, 스크립트에서 환경 변수를 설정할 수 있습니다.

이 코맨드를 실행하세요 `export WANDB_API_KEY=X`, 여기서 X는 자신의 API 키입니다. 로그인된 상태에서 [wandb.ai/authorize](https://app.wandb.ai/authorize)에 방문하여 API 키를 확인할 수 있습니다.