---
title: 커맨드라인 인터페이스
menu:
  reference:
    identifier: ko-ref-cli-_index
---

**사용법**

`wandb [옵션] COMMAND [ARG]...`



**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--version` | 버전을 표시하고 종료합니다. |


**코맨드**

| **코맨드** | **설명** |
| :--- | :--- |
| agent | W&B 에이전트를 실행합니다. |
| artifact | Artifacts 와 상호작용하는 명령어입니다. |
| beta | wandb CLI 명령어의 베타 버전입니다. |
| controller | 로컬 W&B 스윕 컨트롤러를 실행합니다. |
| disabled | W&B 를 비활성화합니다. |
| docker | 코드를 docker 컨테이너에서 실행합니다. |
| docker-run | `docker run` 을 래핑하여 WANDB_API_KEY 와 WANDB_DOCKER...를 추가합니다. |
| enabled | W&B 를 활성화합니다. |
| init | 디렉토리를 Weights & Biases 와 연동 설정합니다. |
| job | W&B job 관리 및 보기 명령어입니다. |
| launch | W&B Job 을 실행하거나 대기열에 추가합니다. |
| launch-agent | W&B launch agent 를 실행합니다. |
| launch-sweep | W&B launch sweep 을 실행합니다 (실험적 기능). |
| login | Weights & Biases 에 로그인합니다. |
| offline | W&B 동기화를 비활성화합니다. |
| online | W&B 동기화를 활성화합니다. |
| pull | Weights & Biases 에서 파일을 가져옵니다. |
| restore | run 을 위한 코드, 설정, docker 상태를 복원합니다. |
| scheduler | W&B launch sweep 스케줄러를 실행합니다 (실험적 기능). |
| server | 로컬 W&B 서버 운영 명령어입니다. |
| status | 설정 정보를 표시합니다. |
| sweep | 하이퍼파라미터 스윕을 초기화합니다. |
| sync | 오프라인 트레이닝 디렉토리를 W&B 에 업로드합니다. |
| verify | 로컬 인스턴스를 검증합니다. |