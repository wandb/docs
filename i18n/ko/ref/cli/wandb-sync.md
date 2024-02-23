
# wandb 동기화

**사용법**

`wandb sync [옵션] [경로]...`

**요약**

오프라인 학습 디렉터리를 W&B에 업로드합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --id | 업로드하고자 하는 실행 ID입니다. |
| -p, --project | 업로드하고자 하는 프로젝트입니다. |
| -e, --entity | 범위를 지정할 엔티티입니다. |
| --job_type | 관련 실행들을 함께 그룹화하기 위한 실행의 유형을 명시합니다. |
| --sync-tensorboard / --no-sync-tensorboard | tfevent 파일들을 wandb로 스트리밍합니다. |
| --include-globs | 포함할 글로브의 쉼표로 구분된 목록입니다. |
| --exclude-globs | 제외할 글로브의 쉼표로 구분된 목록입니다. |
| --include-online / --no-include-online | 온라인 실행 포함 |
| --include-offline / --no-include-offline | 오프라인 실행 포함 |
| --include-synced / --no-include-synced | 동기화된 실행 포함 |
| --mark-synced / --no-mark-synced | 실행을 동기화됨으로 표시 |
| --sync-all | 모든 실행 동기화 |
| --clean | 동기화된 실행 삭제 |
| --clean-old-hours | 이 시간 이전에 생성된 실행 삭제. --clean 플래그와 함께 사용됩니다. |
| --clean-force | 확인 프롬프트 없이 정리합니다. |
| --show | 보여줄 실행의 수 |
| --append | 실행 추가 |
| --skip-console | 콘솔 로그 건너뛰기 |