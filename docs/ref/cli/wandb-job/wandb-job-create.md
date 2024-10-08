# wandb job create

**사용법**

`wandb job create [OPTIONS] {git|code|image} PATH`

**요약**

wandb run 없이 소스에서 job을 생성합니다.

Jobs는 세 가지 유형이 될 수 있습니다: git, code, 또는 image.

git: 경로에 엔트리 포인트가 있거나 명시적으로 제공된 메인 파이썬 실행 파일을 가리키는 git 소스. code: requirements.txt 파일을 포함하는 코드 경로. image: 도커 이미지.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -p, --project | job을 나열하려는 프로젝트. |
| -e, --entity | job이 속한 entity |
| -n, --name | job의 이름 |
| -d, --description | job에 대한 설명 |
| -a, --alias | job의 에일리어스 |
| --entry-point | 실행 가능한 파일 및 entrypoint 파일을 포함하는 스크립트의 엔트리 포인트. code 또는 repo job에 필요합니다. --build-context가 제공되면, 엔트리포인트 코맨드의 경로는 빌드 컨텍스트를 기준으로 하게 됩니다. |
| -g, --git-hash | git job의 소스로 사용할 커밋 참조 |
| -r, --runtime | job을 실행할 파이썬 런타임 |
| -b, --build-context | job의 소스 코드 루트에서 빌드 컨텍스트로의 경로. 제공된 경우, 이것은 Dockerfile과 entrypoint에 대한 기본 경로로 사용됩니다. |
| --base-image | job에 사용할 기본 이미지. image jobs와 호환되지 않음. |
| --dockerfile | job의 Dockerfile 경로. --build-context가 제공되면, Dockerfile 경로는 빌드 컨텍스트를 기준으로 하게 됩니다. |