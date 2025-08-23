---
title: wandb restore
menu:
  reference:
    identifier: ko-ref-cli-wandb-restore
---

**사용법**

`wandb restore [OPTIONS] RUN`

**요약**

해당 run의 코드, config, 도커 상태를 복원합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--no-git` | git 상태를 복원하지 않습니다 |
| `--branch / --no-branch` | 브랜치를 생성할지 아니면 분리된 상태로 체크아웃할지 여부 |
| `-p, --project` | 업로드할 프로젝트 지정 |
| `-e, --entity` | 조회할 Entity 범위 지정 |
