
# 커맨드라인 인터페이스

**사용법**

`wandb [OPTIONS] COMMAND [ARGS]...`


**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --version | 버전을 보여주고 종료합니다. |

**코맨드**

| **코맨드** | **설명** |
| :--- | :--- |
| agent | W&B 에이전트를 실행합니다 |
| artifact | 아티팩트와 상호작용하기 위한 코맨드 |
| beta | wandb CLI 코맨드의 베타 버전입니다. |
| controller | W&B 로컬 스윕 컨트롤러를 실행합니다 |
| disabled | W&B를 비활성화합니다. |
| docker | 코드를 docker 컨테이너에서 실행합니다. |
| docker-run | `docker run`을 감싸고 WANDB_API_KEY와 WANDB_DOCKER...를 추가합니다 |
| enabled | W&B를 활성화합니다. |
| init | Weights & Biases로 디렉토리를 설정합니다 |
| job | W&B 작업을 관리하고 보는 코맨드 |
| launch | W&B 작업을 실행하거나 큐에 넣습니다. |
| launch-agent | W&B 런치 에이전트를 실행합니다. |
| launch-sweep | W&B 런치 스윕을 실행합니다 (실험적). |
| login | Weights & Biases에 로그인합니다 |
| offline | W&B 동기화를 비활성화합니다 |
| online | W&B 동기화를 활성화합니다 |
| pull | Weights & Biases에서 파일을 가져옵니다 |
| restore | 실행을 위한 코드, 설정 및 docker 상태를 복원합니다 |
| scheduler | W&B 런치 스윕 스케줄러를 실행합니다 (실험적) |
| server | 로컬 W&B 서버를 운영하기 위한 코맨드 |
| status | 설정을 보여줍니다 |
| sweep | 하이퍼파라미터 스윕을 초기화합니다. |
| sync | 오프라인 트레이닝 디렉토리를 W&B에 업로드합니다 |
| verify | 로컬 인스턴스를 검증합니다 |