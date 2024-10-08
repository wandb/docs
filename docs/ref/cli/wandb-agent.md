# wandb 에이전트

**사용법**

`wandb agent [OPTIONS] SWEEP_ID`

**요약**

W&B 에이전트 실행

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| -p, --project | W&B run 이 스윕에서 생성되어 보내질 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우, run 은 'Uncategorized'로 레이블된 프로젝트로 전송됩니다. |
| -e, --entity | 스윕에 의해 생성된 W&B run 을 보내고자 하는 사용자명 또는 팀 이름입니다. 지정한 entity가 이미 존재하는지 확인하십시오. entity를 지정하지 않으면, run은 기본 entity인 일반적으로 사용자의 이름으로 전송됩니다. |
| --count | 이 에이전트의 최대 run 횟수입니다. |