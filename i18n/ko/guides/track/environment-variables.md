---
description: Set W&B environment variables.
displayed_sidebar: default
---

# 환경 변수

<head>
  <title>W&B 환경 변수</title>
</head>

자동화된 환경에서 스크립트를 실행할 때, 스크립트가 실행되기 전이나 스크립트 내에서 환경 변수를 설정하여 **wandb**를 제어할 수 있습니다.

```bash
# 이것은 비밀이므로 버전 관리에 체크인해서는 안 됩니다
WANDB_API_KEY=$YOUR_API_KEY
# 이름과 메모는 선택 사항입니다
WANDB_NAME="My first run"
WANDB_NOTES="더 작은 학습률, 더 많은 정규화."
```

```bash
# wandb/settings 파일을 체크인하지 않는 경우에만 필요합니다
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# 스크립트를 클라우드에 동기화하고 싶지 않은 경우
os.environ["WANDB_MODE"] = "offline"
```

## 선택적 환경 변수

원격 머신에서 인증 설정과 같은 작업을 수행하기 위해 이 선택적 환경 변수들을 사용하세요.

| 변수 이름                      | 사용법                                  |
| --------------------------- | ---------- |
| **WANDB\_ANONYMOUS**        | 사용자가 비밀 URL이 있는 익명의 runs을 생성할 수 있도록 "allow", "never", "must" 중 하나로 설정합니다.                                                    |
| **WANDB\_API\_KEY**         | 계정과 연결된 인증 키를 설정합니다. [설정 페이지](https://app.wandb.ai/settings)에서 키를 찾을 수 있습니다. 원격 머신에서 `wandb login`이 실행되지 않았다면 이를 설정해야 합니다.               |
| **WANDB\_BASE\_URL**        | [wandb/local](../hosting/intro.md)을 사용하는 경우 이 환경 변수를 `http://YOUR_IP:YOUR_PORT`로 설정해야 합니다.        |
| **WANDB\_CACHE\_DIR**       | 기본값은 \~/.cache/wandb입니다. 이 위치를 이 환경 변수로 재정의할 수 있습니다.                    |
| **WANDB\_CONFIG\_DIR**      | 기본값은 \~/.config/wandb입니다. 이 위치를 이 환경 변수로 재정의할 수 있습니다.                             |
| **WANDB\_CONFIG\_PATHS**    | wandb.config에 로드할 yaml 파일들의 쉼표로 구분된 목록입니다. [config](./config.md#file-based-configs)을 참조하세요.                                          |
| **WANDB\_CONSOLE**          | "off"로 설정하면 stdout / stderr 로깅을 비활성화합니다. 기본값은 지원하는 환경에서 "on"입니다.                                          |
| **WANDB\_DIR**              | 여기에 생성된 모든 파일을 _wandb_ 디렉토리 대신 절대 경로로 저장하도록 설정합니다. _이 디렉토리가 존재하고 프로세스를 실행하는 사용자가 쓸 수 있도록 해야 합니다_                  |
| **WANDB\_DISABLE\_GIT**     | wandb가 git 저장소를 탐색하고 최신 커밋 / 차이점을 캡처하는 것을 방지합니다.      |
| **WANDB\_DISABLE\_CODE**    | wandb에서 노트북이나 git 차이점을 저장하는 것을 방지하려면 이를 true로 설정하세요. git 저장소에 있다면 여전히 현재 커밋을 저장합니다.                   |
| **WANDB\_DOCKER**           | runs를 복원하는 데 도커 이미지 다이제스트를 설정하세요. 이는 wandb 도커 명령어로 자동으로 설정됩니다. `wandb docker my/image/name:tag --digest`를 실행하여 이미지 다이제스트를 얻을 수 있습니다.    |
| **WANDB\_ENTITY**           | run과 관련된 엔터티입니다. 트레이닝 스크립트의 디렉토리에서 `wandb init`을 실행하면 _wandb_라는 이름의 디렉토리가 생성되고 기본 엔터티가 저장되어 소스 제어에 체크인할 수 있습니다. 파일을 생성하거나 파일을 재정의하고 싶지 않다면 환경 변수를 사용할 수 있습니다. |
| **WANDB\_ERROR\_REPORTING** | wandb가 치명적인 오류를 오류 추적 시스템에 기록하는 것을 방지하려면 이를 false로 설정하세요.                             |
| **WANDB\_HOST**             | 시스템에서 제공한 호스트 이름을 사용하고 싶지 않은 경우 wandb 인터페이스에서 보고 싶은 호스트 이름을 설정하세요.                                |
| **WANDB\_IGNORE\_GLOBS**    | 무시할 파일 글로브의 쉼표로 구분된 목록을 설정하세요. 이 파일들은 클라우드에 동기화되지 않습니다.                              |
| **WANDB\_JOB\_NAME**        | `wandb`에 의해 생성된 모든 작업에 대한 이름을 지정합니다. 자세한 내용은 [작업 생성하기](../launch/create-launch-job.md)를 참조하세요.                                                                                                                                                                                                                        |
| **WANDB\_JOB\_TYPE**        | 작업 유형을 지정하세요. "training"이나 "evaluation"과 같이 다양한 유형의 runs을 나타냅니다. 자세한 내용은 [그룹화](../runs/grouping.md)를 참조하세요.               |
| **WANDB\_MODE**             | 이를 "offline"으로 설정하면 wandb가 run 메타데이터를 로컬에 저장하고 서버에 동기화하지 않습니다. "disabled"로 설정하면 wandb가 완전히 꺼집니다.                  |
| **WANDB\_NAME**             | run의 사람이 읽을 수 있는 이름입니다. 설정하지 않으면 임의로 생성됩니다.                       |
| **WANDB\_NOTEBOOK\_NAME**   | jupyter에서 실행 중이라면 이 변수로 노트북의 이름을 설정할 수 있습니다. 자동 감지를 시도합니다.                    |
| **WANDB\_NOTES**            | run에 대한 긴 메모입니다. 마크다운이 허용되며 나중에 UI에서 이를 편집할 수 있습니다.                                    |
| **WANDB\_PROJECT**          | run과 관련된 프로젝트입니다. 이는 `wandb init`으로도 설정할 수 있지만, 환경 변수가 값을 재정의합니다.                               |
| **WANDB\_RESUME**           | 기본적으로 _never_로 설정됩니다. _auto_로 설정하면 wandb가 실패한 runs를 자동으로 재개합니다. _must_는 run이 시작 시 존재해야 함을 강제합니다. 항상 고유한 ID를 생성하려면 이를 _allow_로 설정하고 항상 **WANDB\_RUN\_ID**를 설정하세요.      |
| **WANDB\_RUN\_GROUP**       | 실험 이름을 지정하여 runs을 자동으로 함께 그룹화합니다. 자세한 내용은 [그룹화](../runs/grouping.md)를 참조하세요.                                 |
| **WANDB\_RUN\_ID**          | 프로젝트당 전역적으로 고유한 문자열(최대 64자)로, 스크립트의 단일 run에 해당합니다. 모든 비단어 문자는 대시로 변환됩니다. 실패의 경우 기존 run을 재개하는 데 사용할 수 있습니다.      |
| **WANDB\_SILENT**           | 이를 **true**로 설정하면 wandb 로그 문장을 무음 처리합니다. 이 설정이 되면 모든 로그가 **WANDB\_DIR**/debug.log에 기록됩니다.               |
| **WANDB\_SHOW\_RUN**        | 이를 **true**로 설정하면 운영 체제가 지원하는 경우 자동으로 run URL을 브라우저에서 열도록 합니다.        |
| **WANDB\_TAGS**             | run에 적용될 태그의 쉼표로 구분된 목록입니다.                 |
| **WANDB\_USERNAME**         | run과 관련된 팀의 멤버 사용자 이름입니다. 서비스 계정 API 키와 함께 사용하여 자동화된 runs을 팀의 멤버에게 할당할 수 있습니다.               |
| **WANDB\_USER\_EMAIL**      | run과 관련된 팀의 멤버 이메일입니다. 서비스 계정 API 키와 함께 사용하여 자동화된 runs을 팀의 멤버에게 할당할 수 있습니다.            |

## Singularity 환경

[Singularity](https://singularity.lbl.gov/index.html)에서 컨테이너를 실행하는 경우 위 변수들을 **SINGULARITYENV\_**로 시작하여 환경 변수를 전달할 수 있습니다. Singularity 환경 변수에 대한 자세한 내용은 [여기](https://singularity.lbl.gov/docs-environment-metadata#environment)에서 찾을 수 있습니다.

## AWS에서 실행하기

AWS에서 배치 작업을 실행하는 경우, [설정 페이지](https://app.wandb.ai/settings)에서 API 키를 가져와 WANDB\_API\_KEY 환경 변수를 [AWS 배치 작업 사양](https://docs.aws.amazon.com/batch/latest/userguide/job\_definition\_parameters.html#parameters)에 설정함으로써 W&B 자격증명으로 머신을 인증할 수 있습니다.

## 자주 묻는 질문

### 자동화된 runs과 서비스 계정

W&B에 로깅하는 runs을 시작하는 자동화된 테스트나 내부 툴이 있으면, 팀 설정 페이지에서 **서비스 계정**을 생성하세요. 이를 통해 자동화된 작업에 서비스 API 키를 사용할 수 있습니다. 서비스 계정 작업을 특정 사용자에게 할당하려면 **WANDB\_USERNAME** 또는 **WANDB\_USER\_EMAIL** 환경 변수를 사용할 수 있습니다.

![자동화된 작업을 위해 팀 설정 페이지에서 서비스 계정 생성](/images/track/common_questions_automate_runs.png)

이는 TravisCI나 CircleCI 같은 툴로 자동화된 단위 테스트를 설정할 때 유용합니다.

### 환경 변수가 wandb.init()에 전달된 파라미터를 덮어씌우나요?

`wandb.init`에 전달된 인수가 환경보다 우선합니다. 환경 변수가 설정되지 않았을 때 시스템 기본값이 아닌 다른 기본값을 사용하려면 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`를 호출할 수 있습니다.

### 로깅 끄기

`wandb offline` 명령은 환경 변수 `WANDB_MODE=offline`을 설정합니다. 이는 기계에서 원격 wandb 서버로의 데이터 동기화를 중지합니다. 여러 프로젝트가 있으면, 모든 프로젝트가 로깅된 데이터를 W&B 서버에 동기화하는 것을 중지합니다.

경고 메시지를 무음 처리하려면:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### 공유 머신에서 여러 wandb 사용자

공유 머신을 사용하고 다른 사람이 wandb 사용자인 경우, runs이 항상 올바른 계정으로 로깅되도록 쉽게 설정할 수 있습니다. WANDB\_API\_KEY 환경 변수를 인증하기 위해 설정하세요. 환경에서 소스를 지정하면 로그인할 때 올바른 자격 증명을 갖게 되거나 스크립트에서 환경 변수를 설정할 수 있습니다.

`export WANDB_API_KEY=X` 명령을 실행하세요. 여기서 X는 API 키입니다. 로그인하면 [wandb.ai/authorize](https://app.wandb.ai/authorize)에서 API 키를 찾을 수 있습니다.