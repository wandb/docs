---
title: wandb sync
menu:
  reference:
    identifier: ko-ref-cli-wandb-sync
---

**사용법**

`wandb sync [OPTIONS] [PATH]...`

**요약**

오프라인 트레이닝 디렉토리를 W&B에 업로드합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--id` | 업로드할 대상 run입니다. |
| `-p, --project` | 업로드할 대상 프로젝트입니다. |
| `-e, --entity` | 범위를 지정할 엔티티입니다. |
| `--job_type` | 관련된 run들을 함께 그룹화하기 위한 run의 유형을 지정합니다. |
| `--sync-tensorboard / --no-sync-tensorboard` | tfevent 파일을 wandb로 스트리밍합니다. |
| `--include-globs` | 포함할 glob 목록 (쉼표로 구분)입니다. |
| `--exclude-globs` | 제외할 glob 목록 (쉼표로 구분)입니다. |
| `--include-online / --no-include-online` | 온라인 run을 포함합니다. |
| `--include-offline / --no-include-offline` | 오프라인 run을 포함합니다. |
| `--include-synced / --no-include-synced` | 동기화된 run을 포함합니다. |
| `--mark-synced / --no-mark-synced` | run을 동기화됨으로 표시합니다. |
| `--sync-all` | 모든 run을 동기화합니다. |
| `--clean` | 동기화된 run을 삭제합니다. |
| `--clean-old-hours` | 지정된 시간보다 먼저 생성된 run을 삭제합니다. --clean 플래그와 함께 사용해야 합니다. |
| `--clean-force` | 확인 프롬프트 없이 삭제합니다. |
| `--show` | 표시할 run의 수입니다. |
| `--append` | run을 추가합니다. |
| `--skip-console` | 콘솔 로그를 건너뜁니다. |
