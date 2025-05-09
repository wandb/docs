---
title: Databricks
description: W&B를 Databricks와 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-databricks
    parent: integrations
weight: 50
---

W&B는 Databricks 환경에서 W&B Jupyter 노트북 경험을 사용자 정의하여 [Databricks](https://www.databricks.com/)와 통합됩니다.

## Databricks 설정

1. 클러스터에 wandb 설치

    클러스터 설정으로 이동하여 클러스터를 선택하고 **Libraries**를 클릭합니다. **Install New**를 클릭하고 **PyPI**를 선택한 다음 `wandb` 패키지를 추가합니다.

2. 인증 설정

    W&B 계정을 인증하려면 노트북이 쿼리할 수 있는 Databricks secret을 추가하면 됩니다.

    ```bash
    # databricks cli 설치
    pip install databricks-cli

    # databricks UI에서 토큰 생성
    databricks configure --token

    # 다음 두 코맨드 중 하나를 사용하여 스코프를 생성합니다 (databricks에서 보안 기능 활성화 여부에 따라 다름).
    # 보안 추가 기능 사용
    databricks secrets create-scope --scope wandb
    # 보안 추가 기능 미사용
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # 다음 위치에서 api_key를 추가합니다: https://app.wandb.ai/authorize
    databricks secrets put --scope wandb --key api_key
    ```

## 예시

### 간단한 예시

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

wandb.init()
wandb.log({"foo": 1})
```

### Sweeps

wandb.sweep() 또는 wandb.agent()를 사용하려는 노트북에 필요한 설정(임시):

```python
import os

# 다음은 향후에는 필요하지 않습니다.
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```