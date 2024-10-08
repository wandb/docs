# wandb artifact put

**사용법**

`wandb artifact put [OPTIONS] PATH`

**요약**

아티팩트를 wandb에 업로드합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -n, --name | 푸시할 아티팩트의 이름:   project/artifact_name |
| -d, --description | 이 아티팩트에 대한 설명 |
| -t, --type | 아티팩트의 유형 |
| -a, --alias | 이 아티팩트에 적용할 에일리어스 |
| --id | 업로드하고 싶은 run. |
| --resume | 현재 디렉토리에서 마지막 run을 재개. |
| --skip_cache | 아티팩트 파일을 업로드하는 동안 캐시를 건너뜁니다. |
| --policy [mutable|immutable] | 아티팩트 파일을 업로드하는 동안 저장 정책을 설정합니다. |