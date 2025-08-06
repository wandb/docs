---
title: wandb job create
menu:
  reference:
    identifier: ko-ref-cli-wandb-job-wandb-job-create
---

**사용법**

`wandb job create [옵션] {git|code|image} PATH`

**요약**

wandb run 없이 소스에서 Job을 생성합니다.

Jobs는 git, code, image 세 가지 타입이 있습니다.

git: 엔트리포인트가 경로에 있거나 명시적으로 지정된 git 소스에서 main python 실행 파일을 가리키는 경우입니다.  
code: requirements.txt 파일이 포함된 코드 경로입니다.  
image: Docker 이미지를 사용합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | Jobs를 조회할 Project를 지정합니다. |
| `-e, --entity` | 해당 Jobs가 속한 Entity를 지정합니다. |
| `-n, --name` | Job의 이름을 지정합니다. |
| `-d, --description` | Job에 대한 설명을 입력합니다. |
| `-a, --alias` | Job의 에일리어스를 설정합니다. |
| `--entry-point` | 실행 파일과 엔트리포인트 파일을 포함하는 스크립트의 엔트리포인트입니다. code 또는 repo Job에서 필수입니다. --build-context가 제공된 경우, 엔트리포인트 코맨드의 경로는 빌드 컨텍스트를 기준으로 상대 경로가 됩니다. |
| `-g, --git-hash` | git Job의 소스로 사용할 커밋 참조입니다.|
| `-r, --runtime` | Job을 실행할 Python 런타임을 지정합니다. |
| `-b, --build-context` | Job 소스 코드의 루트에서 빌드 컨텍스트 경로입니다. 제공되면, Dockerfile과 엔트리포인트의 기준 경로로 사용됩니다. |
| `--base-image` | Job에 사용할 기본 이미지입니다. image Job과는 함께 사용할 수 없습니다. |
| `--dockerfile` | Job에서 사용할 Dockerfile의 경로입니다. --build-context가 있다면, Dockerfile 경로는 빌드 컨텍스트를 기준으로 상대 경로가 됩니다. |