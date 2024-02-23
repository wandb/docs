
# wandb 베타 동기화

**사용법**

`wandb beta sync [OPTIONS] WANDB_DIR`

**요약**

학습 실행을 W&B에 업로드합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --id | 업로드하려는 실행입니다. |
| -p, --project | 업로드하려는 프로젝트입니다. |
| -e, --entity | 범위를 지정할 엔티티입니다. |
| --skip-console | 콘솔 로그를 건너뜁니다. |
| --append | 실행을 추가합니다. |
| -i, --include | 포함할 Glob입니다. 여러 번 사용할 수 있습니다. |
| -e, --exclude | 제외할 Glob입니다. 여러 번 사용할 수 있습니다. |
| --mark-synced / --no-mark-synced | 실행을 동기화된 것으로 표시합니다. |
| --skip-synced / --no-skip-synced | 동기화된 실행을 건너뜁니다. |
| --dry-run | 아무 것도 업로드하지 않고 시험 실행을 수행합니다. |