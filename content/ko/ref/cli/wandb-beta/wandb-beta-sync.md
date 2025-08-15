---
title: wandb beta sync
menu:
  reference:
    identifier: ko-ref-cli-wandb-beta-wandb-beta-sync
---

**사용법**

`wandb beta sync [OPTIONS] WANDB_DIR`

**요약**

트레이닝 run을 W&B에 업로드합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--id` | 업로드할 run을 지정합니다. |
| `-p, --project` | 업로드할 Project를 지정합니다. |
| `-e, --entity` | 범위를 지정할 Entity입니다. |
| `--skip-console` | 콘솔 로그를 건너뜁니다 |
| `--append` | run을 이어서 업로드합니다 |
| `-i, --include` | 포함할 glob 패턴입니다. 여러 번 사용할 수 있습니다. |
| `-e, --exclude` | 제외할 glob 패턴입니다. 여러 번 사용할 수 있습니다. |
| `--mark-synced / --no-mark-synced` | run을 동기화됨으로 표시합니다 |
| `--skip-synced / --no-skip-synced` | 동기화된 run을 건너뜁니다 |
| `--dry-run` | 실제로 아무것도 업로드하지 않고 미리 실행해봅니다. |