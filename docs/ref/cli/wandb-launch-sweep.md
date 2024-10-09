# wandb launch-sweep

**사용법**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**요약**

W&B launch sweep 실행 (실험적).

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -q, --queue | 스윕을 보낼 큐의 이름 |
| -p, --project | 에이전트가 감시할 프로젝트의 이름. 전달되면, 구성 파일을 사용해 전달된 프로젝트 값이 덮어씀 |
| -e, --entity | 사용할 Entity. 기본값은 현재 로그인한 사용자 |
| -r, --resume_id | 8자리 스윕 ID를 전달하여 launch sweep을 재개. 큐 필요 |
| --prior_run | 이 스윕에 추가할 기존 run의 ID |