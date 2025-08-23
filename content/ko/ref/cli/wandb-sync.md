---
title: wandb sync
menu:
  reference:
    identifier: ko-ref-cli-wandb-sync
---

**사용법**

`wandb sync [OPTIONS] [PATH]...`

**요약**

오프라인 트레이닝 디렉토리를 W&B에 업로드합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--id` | 업로드할 run 을 지정합니다. |
| `-p, --project` | 업로드할 Project 를 지정합니다. |
| `-e, --entity` | 적용할 Entity 를 지정합니다. |
| `--job_type` | 관련된 run 들을 함께 그룹화할 run 유형을 지정합니다. |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent 파일을 wandb로 스트리밍합니다. |
| `--include-globs` | 포함할 glob 패턴을 쉼표로 구분해 지정합니다. |
| `--exclude-globs` | 제외할 glob 패턴을 쉼표로 구분해 지정합니다. |
| `--include-online / --no-include-online` | 온라인 run 포함 여부 |
| `--include-offline / --no-include-offline` | 오프라인 run 포함 여부 |
| `--include-synced / --no-include-synced` | 동기화된 run 포함 여부 |
| `--mark-synced / --no-mark-synced` | run 을 동기화됨으로 표시 |
| `--sync-all` | 모든 run 동기화 |
| `--clean` | 동기화된 run 삭제 |
| `--clean-old-hours` | 지정한 시간 이전에 생성된 run 삭제. --clean 플래그와 함께 사용합니다. |
| `--clean-force` | 확인 메시지 없이 강제 삭제합니다. |
| `--show` | 표시할 run 개수 |
| `--append` | run 추가 |
| `--skip-console` | 콘솔 로그 건너뛰기 |