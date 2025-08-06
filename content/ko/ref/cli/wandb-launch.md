---
title: wandb launch
menu:
  reference:
    identifier: ko-ref-cli-wandb-launch
---

**사용법**

`wandb launch [OPTIONS]`

**요약**

W&B Job을 실행하거나 큐에 추가합니다. 자세한 내용은 https://wandb.me/launch 를 참고하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-u, --uri (str)` | 실행할 로컬 경로나 git repo uri를 지정합니다. 이 옵션이 제공되면 해당 uri에서 job을 생성합니다. |
| `-j, --job (str)` | 실행할 job의 이름을 지정합니다. 이 옵션을 사용하면 uri 없이 실행할 수 있습니다. |
| `--entry-point` | 프로젝트 내의 엔트리 포인트입니다. [기본값: main]. 엔트리 포인트를 찾지 못하면 동일한 이름의 프로젝트 파일을 스크립트로 실행합니다. .py 파일은 'python'으로, .sh 파일은 환경 변수 $SHELL로 지정된 기본 shell로 실행됩니다. 이 옵션을 사용하면 config 파일의 entrypoint 값을 덮어씁니다. |
| `--build-context (str)` | 소스 코드 내에서 빌드 컨텍스트의 경로입니다. 기본값은 소스 코드의 루트입니다. -u와만 호환됩니다. |
| `--name` | 실행(run)의 이름을 지정합니다. 지정하지 않으면 임의의 run 이름으로 실행됩니다. 사용 시 config 파일의 name 값을 덮어씁니다. |
| `-e, --entity (str)` | 새 run이 전송될 대상 Entity의 이름입니다. 기본적으로 로컬 wandb/settings 폴더에 설정된 Entity를 사용합니다. 지정하면 config 파일의 entity 값을 덮어씁니다. |
| `-p, --project (str)` | 새 run이 전송될 대상 Project의 이름입니다. 기본적으로 소스 uri에서 주어진 Project 이름이나 github run의 경우 git repo 이름을 사용합니다. 지정하면 config 파일의 project 값을 덮어씁니다. |
| `-r, --resource` | run에 사용할 실행 리소스입니다. 사용 가능한 값: 'local-process', 'local-container', 'kubernetes', 'sagemaker', 'gcp-vertex'. 리소스 설정 없이 큐에 추가 시 필수 파라미터입니다. 지정하면 config 파일의 resource 값을 덮어씁니다. |
| `-d, --docker-image` | 사용할 특정 도커 이미지(형식: name:tag)를 지정합니다. 지정하면 config 파일의 docker image 값을 덮어씁니다. |
| `--base-image` | job 코드를 실행할 도커 이미지를 지정합니다. --docker-image와는 함께 사용할 수 없습니다. |
| `-c, --config` | JSON 파일의 경로('.json'으로 끝나야 함)나 JSON 문자열을 지정할 수 있으며, launch 설정으로 전달됩니다. 실행(run)의 설정 방법을 지정합니다. |
| `-v, --set-var` | Allow listing이 활성화된 큐에서 템플릿 변수 값을 key-value 쌍(`--set-var key1=value1 --set-var key2=value2`)으로 지정합니다. |
| `-q, --queue` | 실행을 전송할 run 큐의 이름을 지정합니다. 지정하지 않으면 단일 run이 바로 실행됩니다. 인수 없이 `--queue`만 사용할 경우 'default' 큐를 기본값으로 사용합니다. 이름을 지정하면, 해당 프로젝트 및 Entity 아래에 해당 이름의 run 큐가 존재해야 합니다. |
| `--async` | job을 비동기로 실행하는 플래그입니다. 기본값은 false로, `--async`를 지정하지 않으면 wandb launch는 job이 끝날 때까지 대기합니다. 이 옵션은 --queue와 호환되지 않으며, 에이전트로 실행 시 비동기 실행 옵션은 wandb launch-agent에서 설정해야 합니다. |
| `--resource-args` | JSON 파일의 경로('.json'으로 끝나야 함)나 JSON 문자열로 컴퓨트 리소스에 전달할 resource 인수로 사용합니다. 백엔드별로 내용이 다르니, 자세한 내용은 문서를 참고하세요. |
| `--dockerfile` | job 빌드에 사용되는 Dockerfile의 경로입니다. job의 루트 기준으로 작성하세요. |
| `--priority [critical|high|medium|low]` | --queue 사용 시, job의 우선 순위를 지정합니다. 우선 순위가 높은 Launch job이 먼저 처리됩니다. 우선 순위는 critical, high, medium, low 순서입니다. |
