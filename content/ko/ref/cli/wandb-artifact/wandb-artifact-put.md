---
title: wandb artifact put
menu:
  reference:
    identifier: ko-ref-cli-wandb-artifact-wandb-artifact-put
---

**사용법**

`wandb artifact put [OPTIONS] PATH`

**요약**

아티팩트를 wandb에 업로드합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-n, --name` | 업로드할 아티팩트의 이름: 프로젝트/artifact_name |
| `-d, --description` | 이 아티팩트에 대한 설명 |
| `-t, --type` | 아티팩트의 유형 |
| `-a, --alias` | 이 아티팩트에 적용할 에일리어스 |
| `--id` | 업로드할 run입니다. |
| `--resume` | 현재 디렉토리에서 마지막 run을 재개합니다. |
| `--skip_cache` | 아티팩트 파일을 업로드하는 동안 캐싱을 건너뜁니다. |
| `--policy [mutable\|immutable]` | 아티팩트 파일을 업로드하는 동안 스토리지 정책을 설정합니다. |
