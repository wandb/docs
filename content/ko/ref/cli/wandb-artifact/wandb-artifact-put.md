---
title: wandb artifact put
menu:
  reference:
    identifier: ko-ref-cli-wandb-artifact-wandb-artifact-put
---

**사용법**

`wandb artifact put [OPTIONS] PATH`

**요약**

artifact 를 wandb 에 업로드합니다.


**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-n, --name` | 업로드할 artifact 의 이름:   project/artifact_name 형식 |
| `-d, --description` | 이 artifact 에 대한 설명 |
| `-t, --type` | artifact 의 타입 |
| `-a, --alias` | 이 artifact 에 적용할 에일리어스 |
| `--id` | 업로드하려는 run |
| `--resume` | 현재 디렉토리에서 마지막 run 을 이어서 실행합니다. |
| `--skip_cache` | artifact 파일을 업로드할 때 캐시를 건너뜁니다. |
| `--policy [mutable|immutable]` | artifact 파일을 업로드할 때 스토리지 정책을 지정합니다. |