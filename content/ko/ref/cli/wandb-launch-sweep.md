---
title: wandb launch-sweep
menu:
  reference:
    identifier: ko-ref-cli-wandb-launch-sweep
---

**사용법**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**요약**

W&B launch 스윕을 실행합니다 (실험적 기능).

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-q, --queue` | 스윕을 푸시할 큐의 이름 |
| `-p, --project` | 에이전트가 감시할 프로젝트 이름. 전달된 경우, 구성 파일을 사용하여 전달된 프로젝트 값을 덮어씁니다. |
| `-e, --entity` | 사용할 엔티티. 기본적으로 현재 로그인한 사용자입니다. |
| `-r, --resume_id` | 8자 스윕 ID를 전달하여 launch 스윕을 재개합니다. 큐가 필요합니다. |
| `--prior_run` | 이 스윕에 추가할 기존 run의 ID |
