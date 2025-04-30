---
title: wandb sweep
menu:
  reference:
    identifier: ko-ref-cli-wandb-sweep
---

**사용법**

`wandb sweep [OPTIONS] CONFIG_YAML_OR_SWEEP_ID`

**요약**

하이퍼파라미터 스윕을 초기화합니다. 다양한 조합을 테스트하여 기계 학습 모델의 비용 함수를 최적화하는 하이퍼파라미터를 검색합니다.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `-p, --project` | 스윕에서 생성된 W&B run이 전송될 **Project** 이름입니다. **Project** 가 지정되지 않은 경우, run은 Uncategorized 라는 **Project** 로 전송됩니다. |
| `-e, --entity` | 스윕에서 생성된 W&B run을 전송할 사용자 이름 또는 팀 이름입니다. 지정한 **Entity** 가 이미 존재하는지 확인하십시오. **Entity** 를 지정하지 않으면 run은 기본 **Entity** (일반적으로 사용자 이름)로 전송됩니다. |
| `--controller` | 로컬 컨트롤러 실행 |
| `--verbose` | 자세한 출력 표시 |
| `--name` | 스윕의 이름입니다. 이름이 지정되지 않은 경우 스윕 ID가 사용됩니다. |
| `--program` | 스윕 프로그램 설정 |
| `--update` | 보류 중인 스윕 업데이트 |
| `--stop` | 새 run 실행을 중지하고 현재 실행 중인 run이 완료되도록 스윕을 종료합니다. |
| `--cancel` | 실행 중인 모든 run을 중단하고 새 run 실행을 중지하도록 스윕을 취소합니다. |
| `--pause` | 새 run 실행을 일시적으로 중지하도록 스윕을 일시 중지합니다. |
| `--resume` | 새 run 실행을 계속하도록 스윕을 재개합니다. |
| `--prior_run` | 이 스윕에 추가할 기존 run의 ID |
