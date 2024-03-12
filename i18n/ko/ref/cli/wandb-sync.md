
# wandb sync

**사용법**

`wandb sync [OPTIONS] [PATH]...`

**요약**

오프라인 트레이닝 디렉토리를 W&B에 업로드합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --id | 업로드하고 싶은 실행을 지정합니다. |
| -p, --project | 업로드하고 싶은 프로젝트를 지정합니다. |
| -e, --entity | 스코프할 엔티티를 지정합니다. |
| --job_type | 관련 실행을 함께 그룹화하기 위한 실행의 유형을 지정합니다. |
| --sync-tensorboard / --no-sync-tensorboard | tfevent 파일을 wandb에 스트리밍합니다. |
| --include-globs | 포함할 글로브의 쉼표로 구분된 목록입니다. |
| --exclude-globs | 제외할 글로브의 쉼표로 구분된 목록입니다. |
| --include-online / --no-include-online | 온라인 실행을 포함합니다 |
| --include-offline / --no-include-offline | 오프라인 실행을 포함합니다 |
| --include-synced / --no-include-synced | 동기화된 실행을 포함합니다 |
| --mark-synced / --no-mark-synced | 실행을 동기화된 것으로 표시합니다 |
| --sync-all | 모든 실행을 동기화합니다 |
| --clean | 동기화된 실행을 삭제합니다 |
| --clean-old-hours | 이 시간보다 오래된 실행을 삭제합니다. --clean 플래그와 함께 사용합니다. |
| --clean-force | 확인 프롬프트 없이 삭제합니다. |
| --show | 표시할 실행의 수 |
| --append | 실행을 추가합니다 |
| --skip-console | 콘솔 로그를 건너뜁니다 |