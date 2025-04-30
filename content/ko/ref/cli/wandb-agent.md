---
title: wandb agent
menu:
  reference:
    identifier: ko-ref-cli-wandb-agent
---

**사용법**

`wandb agent [OPTIONS] SWEEP_ID`

**요약**

W&B 에이전트 실행

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | 스윕에서 생성된 W&B run이 전송될 프로젝트 이름입니다. 프로젝트가 지정되지 않은 경우, run은 'Uncategorized'라는 프로젝트로 전송됩니다. |
| `-e, --entity` | 스윕에서 생성된 W&B run을 전송할 사용자 이름 또는 팀 이름입니다. 지정하는 엔티티가 이미 존재하는지 확인하십시오. 엔티티를 지정하지 않으면, run은 기본 엔티티 (일반적으로 사용자 이름)로 전송됩니다. |
| `--count` | 이 에이전트의 최대 run 횟수입니다. |
