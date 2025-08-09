---
title: wandb launch-sweep
menu:
  reference:
    identifier: ko-ref-cli-wandb-launch-sweep
---

**사용법**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**요약**

W&B Launch Sweep 을 실행합니다 (실험적 기능).


**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-q, --queue` | 스윕을 보낼 queue 의 이름 |
| `-p, --project` | 에이전트가 관찰할 프로젝트 이름. 지정 시, config 파일에서 전달한 프로젝트 값보다 우선 적용됨 |
| `-e, --entity` | 사용할 entity. 기본값은 현재 로그인된 사용자 |
| `-r, --resume_id` | 8글자의 스윕 id 를 전달하여 launch sweep 을 재개. queue 필수 |
| `--prior_run` | 이 스윕에 추가할 기존 run 의 ID |
