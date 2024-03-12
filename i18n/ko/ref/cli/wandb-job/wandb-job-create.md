
# wandb job create

**사용법**

`wandb job create [옵션] {git|code|image} 경로`

**요약**

wandb 실행 없이 소스에서 작업을 생성합니다.

작업에는 git, code, 또는 image의 세 가지 유형이 있습니다.

git: 경로에 있는 또는 명시적으로 제공된 메인 파이썬 실행 파일을 가리키는 git 소스입니다. code: requirements.txt 파일을 포함하는 코드 경로입니다. image: 도커 이미지입니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -p, --project | 작업을 나열하고자 하는 프로젝트입니다. |
| -e, --entity | 작업이 속한 엔티티입니다 |
| -n, --name | 작업의 이름입니다 |
| -d, --description | 작업에 대한 설명입니다 |
| -a, --alias | 작업의 에일리어스입니다 |
| --entry-point | 리포 작업에 필요한 메인 스크립트로의 코드 경로입니다 |
| -g, --git-hash | 특정 git 커밋으로의 해시입니다. |
| -r, --runtime | 작업을 실행할 파이썬 런타임입니다 |