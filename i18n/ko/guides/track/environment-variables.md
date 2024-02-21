---
description: Set W&B environment variables.
displayed_sidebar: default
---

# 환경 변수

<head>
  <title>W&B 환경 변수</title>
</head>

자동화된 환경에서 스크립트를 실행할 때, **wandb**를 환경 변수를 설정하여 스크립트가 실행되기 전이나 스크립트 내에서 제어할 수 있습니다.

```bash
# 이것은 비밀이며 버전 제어에 체크인해서는 안 됩니다
WANDB_API_KEY=$YOUR_API_KEY
# 이름과 노트는 선택사항입니다
WANDB_NAME="내 첫번째 실행"
WANDB_NOTES="학습률을 더 작게, 더 많은 정규화."
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

원격 기계에서 인증을 설정하는 등의 작업을 수행하기 위해 이러한 선택적 환경 변수를 사용하세요.

| 변수 이름                     | 사용법                                  |
| --------------------------- | ---------- |
| **WANDB\_ANONYMOUS**        | 비밀 URL로 익명 실행을 생성할 수 있게 하려면 "allow", "never", "must"로 설정하세요.                                                    |
| **WANDB\_API\_KEY**         | 계정과 연관된 인증 키를 설정합니다. 키는 [설정 페이지](https://app.wandb.ai/settings)에서 찾을 수 있습니다. 원격 기계에서 `wandb login`이 실행되지 않은 경우 반드시 설정해야 합니다.               |
| **WANDB\_BASE\_URL**        | [wandb/local](../hosting/intro.md)을 사용하는 경우 이 환경 변수를 `http://YOUR_IP:YOUR_PORT`로 설정해야 합니다.        |
| **WANDB\_CACHE\_DIR**       | 기본적으로 ~/.cache/wandb입니다. 이 위치를 이 환경 변수로 재정의할 수 있습니다.                    |
| **WANDB\_CONFIG\_DIR**      | 기본적으로 ~/.config/wandb입니다. 이 위치를 이 환경 변수로 재정의할 수 있습니다.                             |
| **WANDB\_CONFIG\_PATHS**    | wandb.config에 로드할 yaml 파일의 쉼표로 구분된 목록입니다. [config](./config.md#file-based-configs)를 참조하세요.                                          |
| **WANDB\_CONSOLE**          | "off"로 설정하여 stdout / stderr 로깅을 비활성화합니다. 환경이 지원하는 경우 기본적으로 "on"으로 설정됩니다.                                          |
| **WANDB\_DIR**              | 생성된 모든 파일을 학습 스크립트와 관련된 _wandb_ 디렉터리 대신 여기에 저장하려면 절대 경로로 설정하세요. _이 디렉터리가 존재하고 프로세스를 실행하는 사용자가 이 디렉터리에 쓸 수 있는지 확인하세요_                  |
| **WANDB\_DISABLE\_GIT**     | wandb가 git 저장소를 찾고 최신 커밋 / 차이를 캡처하지 않도록 합니다.      |
| **WANDB\_DISABLE\_CODE**    | wandb가 노트북이나 git 차이를 저장하지 않도록 하려면 이것을 true로 설정하세요. git 저장소에 있을 경우 여전히 현재 커밋을 저장합니다.                   |
| **WANDB\_DOCKER**           | 실행 복원을 활성화하기 위해 이것을 도커 이미지 다이제스트로 설정하세요. 이는 wandb 도커 명령으로 자동으로 설정됩니다. 이미지 다이제스트는 `wandb docker my/image/name:tag --digest`를 실행하여 얻을 수 있습니다.    |
| **WANDB\_ENTITY**           | 실행과 연관된 엔티티입니다. 학습 스크립트의 디렉터리에서 `wandb init`을 실행하면 _wandb_라는 이름의 디렉터리가 생성되고 기본 엔티티가 저장되어 소스 제어에 체크인될 수 있습니다. 이 파일을 생성하고 싶지 않거나 파일을 덮어쓰고 싶은 경우 환경 변수를 사용할 수 있습니다. |
| **WANDB\_ERROR\_REPORTING** | wandb가 치명적인 오류를 오류 추적 시스템에 로깅하지 않도록 하려면 이것을 false로 설정하세요.                             |
| **WANDB\_HOST**             | 시스템에서 제공하는 호스트 이름을 사용하고 싶지 않은 경우 wandb 인터페이스에서 보고 싶은 호스트 이름으로 설정하세요.                                |
| **WANDB\_IGNORE\_GLOBS**    | 무시할 파일 글롭의 쉼표로 구분된 목록으로 설정하세요. 이 파일들은 클라우드에 동기화되지 않습니다.                              |
| **WANDB\_JOB\_NAME**        | `wandb`에 의해 생성된 모든 작업에 대한 이름을 지정합니다. 자세한 정보는 [작업 생성](../launch/create-launch-job.md)을 참조하세요.                                                                                                                                                                                                                        
| **WANDB\_JOB\_TYPE**        | "training"이나 "evaluation"과 같은 작업 유형을 지정하여 실행의 다른 유형을 나타냅니다. 자세한 정보는 [그룹화](../runs/grouping.md)를 참조하세요.               |
| **WANDB\_MODE**             | 이것을 "offline"으로 설정하면 wandb가 실행 메타데이터를 로컬에 저장하고 서버에 동기화하지 않습니다. 이것을 "disabled"로 설정하면 wandb가 완전히 꺼집니다.                  |
| **WANDB\_NAME**             | 실행의 사람이 읽을 수 있는 이름입니다. 설정되지 않은 경우 임의로 생성됩니다.                       |
| **WANDB\_NOTEBOOK\_NAME**   | jupyter에서 실행 중이라면 이 변수로 노트북의 이름을 설정할 수 있습니다. 우리는 이것을 자동으로 감지하려고 합니다.                    |
| **WANDB\_NOTES**            | 실행에 대한 더 긴 노트입니다. 마크다운이 허용되며 나중에 UI에서 이를 편집할 수 있습니다.                                    |
| **WANDB\_PROJECT**          | 실행과 연관된 프로젝트입니다. 이것은 `wandb init`으로도 설정할 수 있지만, 환경 변수는 값을 덮어씁니다.                               |
| **WANDB\_RESUME**           | 기본적으로 _never_로 설정됩니다. _auto_로 설정하면 wandb가 실패한 실행을 자동으로 이어서 실행합니다. _must_로 설정하면 실행이 시작할 때 존재하도록 강제합니다. 항상 고유한 ID를 생성하고 싶다면 이것을 _allow_로 설정하고 항상 **WANDB\_RUN\_ID**를 설정하세요.      |
| **WANDB\_RUN\_GROUP**       | 실행을 자동으로 함께 그룹화하기 위해 실험 이름을 지정하세요. 자세한 정보는 [그룹화](../runs/grouping.md)를 참조하세요.                                 |
| **WANDB\_RUN\_ID**          | 스크립트의 단일 실행에 해당하는 전 세계적으로 고유한 문자열(프로젝트별)로 설정하세요. 64자를 초과할 수 없습니다. 모든 비단어 문자는 대시로 변환됩니다. 실패의 경우 기존 실행을 이어서 실행하는 데 사용할 수 있습니다.      |
| **WANDB\_SILENT**           | **true**로 설정하여 wandb 로그 문을 무음으로 설정하세요. 이것이 설정되면 모든 로그가 **WANDB\_DIR**/debug.log에 작성됩니다.               |
| **WANDB\_SHOW\_RUN**        | 운영 체제가 지원하는 경우 실행 URL이 있는 브라우저를 자동으로 열도록 **true**로 설정하세요.        |
| **WANDB\_TAGS**             | 실행에 적용될 태그의 쉼표로 구분된 목록입니다.                 |
| **WANDB\_USERNAME**         | 실행과 연관된 팀의 사용자 이름입니다. 이것은 서비스 계정 API 키와 함께 사용하여 자동화된 실행을 팀의 구성원에게 귀속시킬 수 있습니다.               |
| **WANDB\_USER\_EMAIL**      | 실행과 연관된 팀의 구성원의 이메일입니다. 이것은 서비스 계정 API 키와 함께 사용하여 자동화된 실행을 팀의 구성원에게 귀속시킬 수 있습니다.            |

## 싱귤래러티 환경

[Singularity](https://singularity.lbl.gov/index.html)에서 컨테이너를 실행하는 경우 위의 변수 앞에 **SINGULARITYENV\_**를 추가하여 환경 변수를 전달할 수 있습니다. 싱귤래러티 환경 변수에 대한 자세한 내용은 [여기](https://singularity.lbl.gov/docs-environment-metadata#environment)에서 찾을 수 있습니다.

## AWS에서 실행하기

AWS에서 배치 작업을 실행하는 경우, [설정 페이지](https://app.wandb.ai/settings)에서 API 키를 가져와서 [AWS 배치 작업 사양](https://docs.aws.amazon.com/batch/latest/userguide/job\_definition\_parameters.html#parameters)에서 WANDB\_API\_KEY 환경 변수를 설정함으로써 기계를 W&B 자격 증명으로 인증할 수 있습니다.

## 자주 묻는 질문

### 자동화된 실행과 서비스 계정

W&B에 로깅하는 실행을 시작하는 자동화된 테스트나 내부 도구가 있는 경우, 팀 설정 페이지에서 **서비스 계정**을 생성하세요. 이를 통해 자동화된 작업에 대해 서비스 API 키를 사용할 수 있습니다. 서비스 계정 작업을 특정 사용자에게 귀속시키고 싶다면 **WANDB\_USERNAME** 또는 **WANDB\_USER\_EMAIL** 환경 변수를 사용할 수 있습니다.

![자동화된 작업을 위해 팀 설정 페이지에서 서비스 계정을 생성하세요](/images/track/common_questions_automate_runs.png)

이는 TravisCI 또는 CircleCI와 같은 도구를 사용하여 자동화된 단위 테스트를 설정하는 경우 연속 통합에 유용합니다.

### 환경 변수가 wandb.init()에 전달된 파라미터를 덮어씌우나요?

환경보다 `wandb.init`에 전달된 인수가 우선합니다. 환경 변수가 설정되지 않은 경우 시스템 기본값 대신 다른 기본값을 원한다면 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`를 호출할 수 있습니다.

### 로깅 끄기

명령어 `wandb offline`은 환경 변수 `WANDB_MODE=offline`을 설정합니다. 이는 원격 wandb 서버로부터 기계에 동기화되는 모든 데이터를 중지합니다. 여러 프로젝트가 있다면, 모두 W&B 서버로 로그된 데이터 동기화가 중지됩니다.

경고 메시지를 조용히 하려면:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### 공유 기계에서 여러 wandb 사용자

공유 기계를 사용하고 다른 사람이 wandb 사용자인 경우, 실행이 항상 올바른 계정에 로그되도록 하는 것이 쉽습니다. WANDB\_API\_KEY 환경 변수를 인증에 사용하세요. env에서 소스를 실행하면 로그인할 때 올바른 자격 증명을 가지게 되거나 스크립트에서 환경 변수를 설정할 수 있습니다.

이 명령어를 실행하세요 `export WANDB_API_KEY=X` 여기서 X는 당신의 API 키입니다. 로그인되어 있다면 [wandb.ai/authorize](https://app.wandb.ai/authorize)에서 API 키를 찾을 수 있습니다.