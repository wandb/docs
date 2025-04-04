---
title: wandb launch
menu:
  reference:
    identifier: ko-ref-cli-wandb-launch
---

**사용법**

`wandb launch [OPTIONS]`

**요약**

W&B Job을 시작하거나 대기열에 추가합니다. https://wandb.me/launch를 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-u, --uri (str)` | 시작할 로컬 경로 또는 Git 저장소 URI입니다. 제공된 경우 이 코맨드는 지정된 URI에서 Job을 생성합니다. |
| `-j, --job (str)` | 시작할 Job의 이름입니다. 전달된 경우 Launch는 URI를 필요로 하지 않습니다. |
| `--entry-point` | 프로젝트 내 진입점입니다. [기본값: main]. 진입점을 찾을 수 없는 경우, 지정된 이름으로 프로젝트 파일을 스크립트로 실행하려고 시도합니다. `.py` 파일은 'python'을 사용하여 실행하고, `.sh` 파일은 환경 변수 $SHELL에 지정된 기본 셸을 사용하여 실행합니다. 전달된 경우, 설정 파일을 사용하여 전달된 진입점 값을 덮어씁니다. |
| `--build-context (str)` | 소스 코드 내 빌드 컨텍스트 경로입니다. 기본값은 소스 코드의 루트입니다. -u와만 호환됩니다. |
| `--name` | run을 시작할 run 이름입니다. 지정하지 않으면 임의의 run 이름이 run을 시작하는 데 사용됩니다. 전달된 경우, 설정 파일을 사용하여 전달된 이름을 덮어씁니다. |
| `-e, --entity (str)` | 새 run이 전송될 대상 Entity의 이름입니다. 기본적으로 로컬 wandb/settings 폴더에 의해 설정된 Entity를 사용합니다. 전달된 경우, 설정 파일을 사용하여 전달된 Entity 값을 덮어씁니다. |
| `-p, --project (str)` | 새 run이 전송될 대상 Project의 이름입니다. 기본적으로 소스 URI에서 제공하는 Project 이름 또는 Github run의 경우 Git 저장소 이름을 사용합니다. 전달된 경우, 설정 파일을 사용하여 전달된 Project 값을 덮어씁니다. |
| `-r, --resource` | run에 사용할 실행 리소스입니다. 지원되는 값: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex' 입니다. 리소스 설정 없이 대기열에 푸시하는 경우 이제 필수 파라미터입니다. 전달된 경우, 설정 파일을 사용하여 전달된 리소스 값을 덮어씁니다. |
| `-d, --docker-image` | 사용하려는 특정 Docker 이미지입니다. name:tag 형식입니다. 전달된 경우, 설정 파일을 사용하여 전달된 Docker 이미지 값을 덮어씁니다. |
| `--base-image` | Job 코드를 실행할 Docker 이미지입니다. --docker-image와 호환되지 않습니다. |
| `-c, --config` | Launch 설정을 전달할 JSON 파일(확장자가 '.json'이어야 함) 경로 또는 JSON 문자열입니다. Launch된 run이 구성되는 방식을 지정합니다. |
| `-v, --set-var` | 허용 목록이 활성화된 대기열에 대한 템플릿 변수 값을 설정합니다. 키-값 쌍으로 지정합니다. 예: `--set-var key1=value1 --set-var key2=value2` |
| `-q, --queue` | 푸시할 run 대기열의 이름입니다. 없는 경우 단일 run을 직접 시작합니다. 인수 없이 제공된 경우 (`--queue`), 기본적으로 'default' 대기열로 설정됩니다. 그렇지 않고 이름이 제공된 경우, 지정된 run 대기열은 제공된 Project 및 Entity 하에 존재해야 합니다. |
| `--async` | Job을 비동기적으로 실행하는 플래그입니다. 기본값은 false입니다. 즉, --async가 설정되지 않은 경우 wandb launch는 Job이 완료될 때까지 기다립니다. 이 옵션은 --queue와 호환되지 않습니다. 에이전트로 실행할 때 비동기 옵션은 wandb launch-agent에서 설정해야 합니다. |
| `--resource-args` | 컴퓨팅 리소스에 리소스 인수로 전달될 JSON 파일(확장자가 '.json'이어야 함) 경로 또는 JSON 문자열입니다. 제공해야 하는 정확한 콘텐츠는 각 실행 백엔드마다 다릅니다. 이 파일의 레이아웃은 설명서를 참조하십시오. |
| `--dockerfile` | Job을 빌드하는 데 사용되는 Dockerfile의 경로입니다 (Job의 루트를 기준으로). |
| `--priority [critical|high|medium|low]` | --queue가 전달되면 Job의 우선 순위를 설정합니다. 우선 순위가 높은 Launch Job이 먼저 처리됩니다. 가장 높은 우선 순위부터 가장 낮은 우선 순위 순서는 다음과 같습니다: critical, high, medium, low |
