# wandb restore

**사용법**

`wandb restore [OPTIONS] RUN`

**요약**

run의 코드, 설정, 도커 상태를 복원합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --no-git | git 상태를 복원하지 않음 |
| --branch / --no-branch | 브랜치를 생성할지 또는 분리된 상태로 체크아웃할지 여부 |
| -p, --project | 업로드하고 싶은 프로젝트 |
| -e, --entity | 목록을 범위로 제한할 엔터티 |
