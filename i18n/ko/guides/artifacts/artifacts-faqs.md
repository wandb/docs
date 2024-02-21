---
description: Answers to frequently asked question about W&B Artifacts.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 아티팩트 FAQ

<head>
  <title>아티팩트에 대해 자주 묻는 질문</title>
</head>

다음 질문들은 [W&B 아티팩트](#questions-about-artifacts) 및 [W&B 아티팩트 워크플로](#questions-about-artifacts-workflows)에 대해 자주 묻는 질문들입니다.

## 아티팩트에 대한 질문들

### 내 아티팩트를 누가 엑세스할 수 있나요?

아티팩트는 상위 프로젝트의 엑세스 권한을 상속받습니다:

* 프로젝트가 비공개인 경우, 프로젝트 팀의 구성원만이 아티팩트에 엑세스할 수 있습니다.
* 공개 프로젝트의 경우, 모든 사용자가 아티팩트를 읽을 수 있지만 프로젝트 팀의 구성원만이 아티팩트를 생성하거나 수정할 수 있습니다.
* 오픈 프로젝트의 경우, 모든 사용자가 아티팩트에 읽고 쓰기 엑세스 권한을 가집니다.

## 아티팩트 워크플로에 대한 질문들

이 섹션은 아티팩트를 관리하고 편집하기 위한 워크플로에 대해 설명합니다. 이러한 워크플로 중 많은 부분이 [W&B API](../track/public-api-guide.md), W&B에 저장된 데이터에 엑세스할 수 있는 [우리 클라이언트 라이브러리](../../ref/python/README.md)의 구성 요소를 사용합니다.

### 기존 실행에 아티팩트를 어떻게 로그하나요?

가끔, 이전에 로그된 실행의 출력으로 아티팩트를 표시하고 싶을 수도 있습니다. 그 시나리오에서는 다음과 같이 [이전 실행을 재초기화](../runs/resuming.md)하고 새로운 아티팩트를 로그할 수 있습니다:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### 아티팩트에 보존 또는 만료 정책을 어떻게 설정하나요?

PII와 같은 개인 정보 보호 규정을 준수해야 하는 데이터세트 아티팩트 또는 저장공간을 관리하기 위해 아티팩트 버전의 삭제를 예약하고자 하는 경우 TTL(생존 시간) 정책을 설정할 수 있습니다. [이 가이드](./ttl.md)에서 자세히 알아보세요.

### 실행에 로그된 아티팩트나 사용된 아티팩트를 어떻게 찾나요? 아티팩트를 생성하거나 사용한 실행을 어떻게 찾나요?

W&B는 주어진 실행이 로그한 아티팩트와 주어진 실행이 사용한 아티팩트를 자동으로 추적하고 이 정보를 사용하여 아티팩트 그래프를 구성합니다 -- 실행과 아티팩트가 노드인 이분, 방향성, 비순환 그래프로, [이것](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)처럼 보입니다("Explode"를 클릭하여 전체 그래프를 보세요).

[Public API](../../ref/python/public-api/README.md)를 사용하여 실행이나 아티팩트에서 시작하여 이 그래프를 프로그래밍 방식으로 탐색할 수 있습니다.

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: '아티팩트에서', value: 'from_artifact'},
    {label: '실행에서', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# 아티팩트에서 그래프를 위로 탐색:
producer_run = artifact.logged_by()
# 아티팩트에서 그래프를 아래로 탐색:
consumer_runs = artifact.used_by()

# 실행에서 그래프를 아래로 탐색:
next_artifacts = consumer_runs[0].logged_artifacts()
# 실행에서 그래프를 위로 탐색:
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# 실행에서 그래프를 아래로 탐색:
produced_artifacts = run.logged_artifacts()
# 실행에서 그래프를 위로 탐색:
consumed_artifacts = run.used_artifacts()

# 아티팩트에서 그래프를 위로 탐색:
earlier_run = consumed_artifacts[0].logged_by()
# 아티팩트에서 그래프를 아래로 탐색:
consumer_runs = produced_artifacts[0].used_by()
```

  </TabItem>
</Tabs>

### 스윕에서 실행된 모델을 어떻게 최적으로 로그하나요?

스윕에서 모델을 로그하는 효과적인 패턴 중 하나는 스윕의 모델 아티팩트를 가지고 있으며, 버전이 스윕에서 다른 실행에 해당하는 것입니다. 더 구체적으로는 다음과 같이 할 수 있습니다:

```python
wandb.Artifact(name="sweep_name", type="model")
```

### 스윕에서 가장 좋은 실행의 아티팩트를 어떻게 찾나요?

스윕에서 가장 성능이 좋은 실행과 관련된 아티팩트를 검색하기 위해 다음 코드를 사용할 수 있습니다:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```

### 코드를 어떻게 저장하나요?‌

실행을 시작하는 주 스크립트나 노트북을 저장하기 위해 `wandb.init`에서 `save_code=True`를 사용하세요. 모든 코드를 실행에 저장하려면 아티팩트와 함께 코드를 버전 관리하세요. 예를 들면 다음과 같습니다:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```

### 여러 아키텍처와 실행을 사용하는 아티팩트는 어떻게 사용하나요?

모델을 _버전 관리_하는 방법에 대해 생각할 수 있는 여러 가지 방법이 있습니다. 아티팩트는 모델 버전 관리를 적절하게 구현할 수 있는 도구를 제공합니다. 여러 실행에서 다양한 모델 아키텍처를 탐색하는 프로젝트에 대한 일반적인 패턴은 아키텍처별로 아티팩트를 분리하는 것입니다. 예를 들면 다음과 같습니다:

1. 각기 다른 모델 아키텍처에 대해 새로운 아티팩트를 생성합니다. 아티팩트의 `metadata` 속성을 사용하여 아키텍처를 더 자세히 설명할 수 있습니다(실행에 `config`를 사용하는 것과 유사하게).
2. 각 모델에 대해 주기적으로 `log_artifact`를 사용하여 체크포인트를 로그합니다. W&B는 이러한 체크포인트의 기록을 자동으로 구축하며, 가장 최근의 체크포인트를 `latest` 별칭으로 표시하여 주어진 모델 아키텍처의 최신 체크포인트를 `architecture-name:latest`를 사용하여 참조할 수 있습니다.

## 참조 아티팩트 FAQ

### W&B에서 이 버전 ID와 ETag를 어떻게 가져올 수 있나요?

W&B에 아티팩트 참조를 로그하고 버킷에 버전 관리가 활성화된 경우 S3 UI에서 버전 ID를 볼 수 있습니다. W&B에서 이러한 버전 ID와 ETag를 가져오려면 아티팩트를 가져온 다음 해당 매니페스트 항목을 가져옵니다. 예를 들면 다음과 같습니다:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```