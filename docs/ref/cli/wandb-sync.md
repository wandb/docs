# wandb sync

**사용법**

`wandb sync [OPTIONS] [PATH]...`

**요약**

오프라인 트레이닝 디렉토리를 W&B에 업로드합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --id | 업로드하려는 run의 ID입니다. |
| -p, --project | 업로드하려는 프로젝트입니다. |
| -e, --entity | 범위를 지정할 entity입니다. |
| --job_type | 관련된 runs를 그룹화하기 위한 run의 유형을 지정합니다. |
| --sync-tensorboard / --no-sync-tensorboard | tfevent 파일을 wandb로 스트리밍합니다. |
| --include-globs | 포함할 globs의 쉼표로 구분된 목록입니다. |
| --exclude-globs | 제외할 globs의 쉼표로 구분된 목록입니다. |
| --include-online / --no-include-online | 온라인 runs 포함 |
| --include-offline / --no-include-offline | 오프라인 runs 포함 |
| --include-synced / --no-include-synced | 동기화된 runs 포함 |
| --mark-synced / --no-mark-synced | runs를 동기화된 것으로 표시 |
| --sync-all | 모든 runs 동기화 |
| --clean | 동기화된 runs 삭제 |
| --clean-old-hours | 이 시간보다 이전에 생성된 runs를 삭제합니다. --clean 플래그와 함께 사용해야 합니다. |
| --clean-force | 확인 프롬프트 없이 정리합니다. |
| --show | 표시할 runs의 수 |
| --append | run 추가 |
| --skip-console | 콘솔 로그 생략 |