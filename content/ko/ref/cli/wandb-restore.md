---
title: wandb restore
menu:
  reference:
    identifier: ko-ref-cli-wandb-restore
---

**사용법**

`wandb restore [OPTIONS] RUN`

**요약**

run에 대한 코드, config, Docker 상태 복원


**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--no-git` | Git 상태를 복원하지 않습니다. |
| `--branch / --no-branch` | branch를 만들지 분리된 checkout을 할지 여부 |
| `-p, --project` | 업로드하려는 project입니다. |
| `-e, --entity` | 목록의 범위를 지정할 entity입니다. |
