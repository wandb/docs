---
title: Artifact FAQ
description: W&B Artifacts에 대한 자주 묻는 질문에 대한 답변.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Artifacts 및 [W&B Artifact 워크플로우](#questions-about-artifacts-workflows)에 대한 자주 묻는 질문입니다.

## Artifacts에 대한 질문

### 누구에게 내 아티팩트에 대한 엑세스가 있나요?

Artifacts는 상위 프로젝트의 엑세스를 상속받습니다:

* 프로젝트가 비공개인 경우, 프로젝트 팀의 멤버들만 아티팩트에 엑세스할 수 있습니다.
* 공개 프로젝트의 경우, 모든 사용자가 아티팩트를 읽을 수 있는 엑세스 권한을 가지지만, 프로젝트 팀의 멤버들만 아티팩트를 생성하거나 수정할 수 있습니다.
* 개방 프로젝트의 경우, 모든 사용자가 아티팩트를 읽고 쓸 수 있는 엑세스 권한을 가집니다.

## Artifacts 워크플로우에 대한 질문

이 섹션에서는 Artifacts를 관리하고 편집하는 워크플로우를 설명합니다. 이러한 워크플로우 중 다수는 [W&B API](../track/public-api-guide.md)를 사용하며, 이는 W&B에 저장된 데이터에 엑세스할 수 있는 [클라이언트 라이브러리](../../ref/python/README.md)의 구성 요소입니다.

### 기존 Run에 아티팩트를 로그하려면 어떻게 하나요?

때때로 이전에 로그된 run의 출력물로 아티팩트를 표시하고 싶을 수 있습니다. 이 경우, [이전 run을 재초기화](../runs/resuming.md)하고 새로운 아티팩트를 다음과 같이 로그할 수 있습니다:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### 내 아티팩트에 보존 또는 만료 정책을 설정하려면 어떻게 하나요?

PII를 포함하는 데이터셋 아티팩트와 같은 데이터 개인정보 보호 규정을 준수해야 하는 아티팩트가 있거나 스토리지를 관리하기 위해 아티팩트 버전의 삭제를 예약하고자 하는 경우, TTL(생존 시간) 정책을 설정할 수 있습니다. [이 가이드](./ttl.md)에서 자세히 알아보세요.

### Run에 의해 로그되거나 소비된 아티팩트를 어떻게 찾을 수 있나요? 아티팩트를 생성하거나 소비한 run을 어떻게 찾을 수 있나요?

W&B는 주어진 run에 의해 로그된 아티팩트와 사용된 아티팩트를 자동으로 추적하며, 이 정보를 사용하여 주어진 run의 아티팩트 그래프, 즉 이분 그래프, 방향성 그래프, 비순환 그래프를 구성합니다. 각각의 노드는 run과 아티팩트로 구성되며, 예시는 [여기](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)를 참고하세요 (전체 그래프를 보려면 "Explode"를 클릭).

이 그래프는 [Public API](../../ref/python/public-api/README.md)를 사용하여 programmatically으로 탐색할 수 있으며, run 또는 아티팩트에서 시작할 수 있습니다.

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: 'From an Artifact', value: 'from_artifact'},
    {label: 'From a Run', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# 아티팩트에서 그래프 위로 이동:
producer_run = artifact.logged_by()
# 아티팩트에서 그래프 아래로 이동:
consumer_runs = artifact.used_by()

# run에서 그래프 아래로 이동:
next_artifacts = consumer_runs[0].logged_artifacts()
# run에서 그래프 위로 이동:
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run에서 그래프 아래로 이동:
produced_artifacts = run.logged_artifacts()
# run에서 그래프 위로 이동:
consumed_artifacts = run.used_artifacts()

# 아티팩트에서 그래프 위로 이동:
earlier_run = consumed_artifacts[0].logged_by()
# 아티팩트에서 그래프 아래로 이동:
consumer_runs = produced_artifacts[0].used_by()
```

  </TabItem>
</Tabs>

### 스윕에서 Run으로부터 모델을 최적으로 로그하려면 어떻게 해야 하나요?

[sweep](../sweeps/intro.md)에서 모델을 로그하는 효과적인 패턴 중 하나는 스윕에 대한 모델 아티팩트를 생성하는 것입니다. 여기서 버전은 스윕의 다양한 run에 해당합니다. 더 구체적으로:

```python
wandb.Artifact(name="sweep_name", type="model")
```

### 스윕에서 최고의 run의 아티팩트를 어떻게 찾을 수 있나요?

스윕에서 최고 성능을 보인 run에 연결된 아티팩트를 검색하려면 다음 코드를 사용할 수 있습니다:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```

### 코드를 어떻게 저장하나요?

`wandb.init`에서 `save_code=True`를 사용하여 run을 실행하는 메인 스크립트 또는 노트북을 저장합니다. 모든 코드를 run에 저장하려면, Artifacts로 코드의 버전을 관리합니다. 예시는 다음과 같습니다:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```

### 여러 아키텍처와 run에서 아티팩트를 사용하는 방법?

모델을 _버전_하는 방법에는 다양한 방식이 있습니다. Artifacts는 모델 버전 관리를 원하는 방식으로 구현할 수 있는 툴을 제공합니다. 여러 모델 아키텍처를 여러 run에서 탐색하는 프로젝트의 일반적인 패턴은 아키텍처로 아티팩트를 분리하는 것입니다. 예를 들어 다음과 같이 할 수 있습니다:

1. 각기 다른 모델 아키텍처마다 새로운 아티팩트를 생성합니다. 아티팩트의 `메타데이터` 속성을 사용하여 아키텍처를 더 상세히 설명할 수 있습니다 (run의 `설정`을 사용하는 것과 유사합니다).
2. 각 모델에 대해 주기적으로 체크포인트를 `log_artifact`로 기록합니다. W&B는 자동으로 해당 체크포인트의 히스토리를 작성하고, 가장 최근의 체크포인트를 `latest` 에일리어스로 주석 처리하여, 주어진 모델 아키텍처에 대한 가장 최신 체크포인트를 `architecture-name:latest`로 참조할 수 있도록 합니다.

## 레퍼런스 아티팩트 FAQ

### W&B에서 이 버전 ID와 ETag를 어떻게 가져올 수 있나요?

W&B와 함께 아티팩트 레퍼런스를 로그했고, 버전 관리가 버킷에서 활성화되어 있다면, 버전 ID는 S3 UI에서 확인할 수 있습니다. W&B에서 이 버전 ID와 ETag를 가져오려면 아티팩트를 가져오고 해당하는 매니페스트 항목을 얻으면 됩니다. 예를 들어:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```