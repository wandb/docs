---
title: wandb launch-agent
menu:
  reference:
    identifier: ko-ref-cli-wandb-launch-agent
---

**사용법**

`wandb launch-agent [OPTIONS]`

**요약**

W&B Launch 에이전트를 실행합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-q, --queue` | 에이전트가 감시할 큐의 이름입니다. 여러 개의 -q 플래그를 사용할 수 있습니다. |
| `-e, --entity` | 사용할 Entity 입니다. 기본값은 현재 로그인된 사용자입니다. |
| `-l, --log-file` | 에이전트 내부 로그의 저장 위치입니다. `-` 를 사용하면 stdout으로 출력합니다. 기본적으로 모든 에이전트 로그는 wandb/ 하위 디렉토리의 debug.log 파일 또는 WANDB_DIR이 설정되어 있다면 해당 위치로 저장됩니다. |
| `-j, --max-jobs` | 이 에이전트가 동시에 실행할 수 있는 Launch 작업의 최대 개수입니다. 기본값은 1입니다. 제한을 두지 않으려면 -1로 설정하세요. |
| `-c, --config` | 사용할 에이전트 config yaml 파일 경로입니다. |
| `-v, --verbose` | 자세한 출력을 표시합니다. |