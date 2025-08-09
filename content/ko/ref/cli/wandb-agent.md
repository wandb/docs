---
title: wandb 에이전트
menu:
  reference:
    identifier: ko-ref-cli-wandb-agent
---

**사용법**

`wandb agent [옵션] SWEEP_ID`

**요약**

W&B 에이전트를 실행합니다

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | 스윕에서 생성된 W&B run 이 전송될 프로젝트 이름입니다. 프로젝트를 지정하지 않으면 run 은 'Uncategorized' 라는 프로젝트로 전송됩니다. |
| `-e, --entity` | 스윕으로 생성된 W&B run 을 전송할 사용자명 또는 팀 이름입니다. 해당 entity 가 이미 존재하는지 확인하세요. 지정하지 않으면 기본 entity(보통 본인 사용자명)로 run 이 전송됩니다. |
| `--count` | 이 에이전트로 실행할 run 의 최대 개수입니다. |