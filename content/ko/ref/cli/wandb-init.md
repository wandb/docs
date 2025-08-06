---
title: wandb init
menu:
  reference:
    identifier: ko-ref-cli-wandb-init
---

**사용법**

`wandb init [OPTIONS]`

**요약**

디렉토리를 Weights & Biases로 설정합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | 사용할 Project를 지정합니다. |
| `-e, --entity` | Project의 범위를 지정할 Entity를 선택합니다. |
| `--reset` | 설정을 초기화합니다. |
| `-m, --mode` | "online", "offline", "disabled" 중에서 선택할 수 있습니다. 기본값은 online입니다. |