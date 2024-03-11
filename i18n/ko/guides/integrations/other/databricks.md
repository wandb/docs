---
description: How to integrate W&B with Databricks.
slug: /guides/integrations/databricks
displayed_sidebar: default
---

# Databricks

W&B는 Databricks 환경에서 W&B 노트북 경험을 맞춤 설정함으로써 [Databricks](https://www.databricks.com/)와 통합됩니다.

## Databricks 설정

### 클러스터에 wandb 설치하기

클러스터 설정으로 이동하여, 클러스터를 선택한 다음, Libraries를 클릭하고, Install New를 선택한 후, PyPI에서 `wandb` 패키지를 추가합니다.

### 인증

W&B 계정을 인증하기 위해, 노트북이 조회할 수 있는 databricks 비밀 정보를 추가할 수 있습니다.

```bash
# databricks cli 설치
pip install databricks-cli

# databricks UI에서 토큰 생성
databricks configure --token

# 두 명령어 중 하나를 사용하여 scope 생성 (databricks에 보안 기능이 활성화되어 있는지에 따라 다름):
# 보안 애드온이 있는 경우
databricks secrets create-scope --scope wandb
# 보안 애드온이 없는 경우
databricks secrets create-scope --scope wandb --initial-manage-principal users

# https://app.wandb.ai/authorize 에서 api_key 추가
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

wandb.sweep() 또는 wandb.agent()를 사용하려는 노트북에 필요한 설정 (임시):

```python
import os

# 이러한 설정은 미래에 필요하지 않게 될 것입니다
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```