---
title: API 가이드
menu:
  reference:
    identifier: ko-ref-python-python_api_walkthrough
weight: 1
---

다양한 W&B API를 언제, 어떻게 사용하여 머신러닝 워크플로우에서 모델 아티팩트를 추적, 공유 및 관리할 수 있는지 알아보세요. 이 페이지에서는 실험 로그 저장, 리포트 생성, 각 작업에 적합한 W&B API를 사용하여 저장된 데이터에 엑세스하는 방법을 다룹니다.

W&B는 다음과 같은 API를 제공합니다:

* W&B Python SDK (`wandb.sdk`): 트레이닝 중 실험 로그 및 모니터링.
* W&B Public API (`wandb.apis.public`): 저장된 실험 데이터 쿼리 및 분석.
* W&B Report and Workspace API (`wandb.wandb-workspaces`): 발견한 내용을 요약하는 리포트 생성.

## 회원가입 및 API 키 생성
W&B에 머신을 인증하려면 먼저 [wandb.ai/authorize](https://wandb.ai/authorize)에서 API 키를 생성해야 합니다. API 키를 복사해 안전하게 보관하세요.

## 패키지 설치 및 임포트

이 가이드에 필요한 W&B 라이브러리와 기타 패키지를 설치하세요.  

```python
pip install wandb
```

W&B Python SDK를 임포트합니다:

```python
import wandb
```

다음 코드 블록에서 팀의 entity를 지정합니다:

```python
TEAM_ENTITY = "<Team_Entity>" # 이 부분을 팀 entity로 교체하세요
PROJECT = "my-awesome-project"
```

## 모델 트레이닝

다음 코드는 기본 머신러닝 워크플로우를 시뮬레이션합니다: 모델 트레이닝, 메트릭 로그, 그리고 모델을 artifact로 저장합니다.

트레이닝 중 W&B와 상호작용하려면 W&B Python SDK (`wandb.sdk`)를 사용하세요. [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ko" >}})로 loss를 로그하고, 트레이닝된 모델을 [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}})로 artifact로 저장한 뒤 [`Artifact.add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ko" >}})를 이용해 모델 파일을 추가합니다.

```python
import random # 데이터 시뮬레이션용

def model(training_data: int) -> int:
    """데모용 모델 시뮬레이션"""
    return training_data * 2 + random.randint(-1, 1)  

# 가중치와 노이즈 시뮬레이션
weights = random.random() # 랜덤 가중치 초기화
noise = random.random() / 5  # 작은 랜덤 노이즈

# 하이퍼파라미터 및 설정
config = {
    "epochs": 10,  # 트레이닝할 에포크 수
    "learning_rate": 0.01,  # 옵티마이저의 학습률
}

# with 컨텍스트 매니저를 사용해 W&B run 시작/종료
with wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config) as run:    
    # 트레이닝 루프 시뮬레이션
    for epoch in range(config["epochs"]):
        xb = weights + noise  # 입력 트레이닝 데이터 시뮬레이션
        yb = weights + noise * 2  # 타겟 출력값 시뮬레이션 (입력 노이즈의 두 배)
        
        y_pred = model(xb)  # 모델 예측값
        loss = (yb - y_pred) ** 2  # 평균제곱오차(MSE) 손실

        print(f"epoch={epoch}, loss={y_pred}")
        # 에포크와 손실을 W&B에 로그
        run.log({
            "epoch": epoch,
            "loss": loss,
        })

    # 모델 artifact를 위한 고유 이름
    model_artifact_name = f"model-demo"  

    # 모델 파일을 저장할 로컬 경로
    PATH = "model.txt" 

    # 모델을 로컬에 저장
    with open(PATH, "w") as f:
        f.write(str(weights)) # 모델 가중치를 파일로 저장

    # artifact 오브젝트 생성
    # 로컬에 저장한 모델 파일을 artifact에 추가
    artifact = wandb.Artifact(name=model_artifact_name, type="model", description="My trained model")
    artifact.add_file(local_path=PATH)
    artifact.save()
```

위 코드 블록의 주요 포인트는 다음과 같습니다:
* 트레이닝 중 메트릭 로그에는 `wandb.Run.log()`를 사용하세요.
* 모델(또는 데이터셋 등)을 Artifact로 W&B 프로젝트에 저장하려면 `wandb.Artifact`를 사용하세요.

이제 모델을 트레이닝하고 Artifact로 저장했으니, 이를 W&B의 registry에 게시할 수 있습니다. [`wandb.Run.use_artifact()`]({{< relref path="/ref/python/sdk/classes/run/#method-runuse_artifact" lang="ko" >}})를 사용해 프로젝트에서 Artifact를 가져오고 모델 레지스트리에 게시할 준비를 합니다. `wandb.Run.use_artifact()`의 주요 목적:
* 프로젝트에서 artifact 오브젝트를 가져옴
* 해당 artifact를 run의 입력으로 표시하여 재현성과 추적성을 보장함. 자세한 내용은 [계보 맵 생성 및 보기]({{< relref path="/guides/core/registry/lineage/" lang="ko" >}})를 참고하세요.

## 모델을 Model registry에 게시하기

조직 내 다른 사람들과 모델을 공유하려면, `wandb.Run.link_artifact()`를 사용해 [collection]({{< relref path="/guides/core/registry/create_collection" lang="ko" >}})에 게시하세요. 아래 코드는 Artifact를 [core Model registry]({{< relref path="/guides/core/registry/registry_types/#core-registry" lang="ko" >}})에 연결하여 팀이 엑세스할 수 있게 합니다.

```python
# Artifact name은 팀 프로젝트 내 특정 artifact 버전을 지정합니다.
artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
print("Artifact name: ", artifact_name)

REGISTRY_NAME = "Model" # W&B 내 registry의 이름
COLLECTION_NAME = "DemoModels"  # 레지스트리 내 컬렉션 이름

# 레지스트리 내 artifact의 타깃 경로 생성
target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
print("Target path: ", target_path)

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
model_artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
run.link_artifact(artifact=model_artifact, target_path=target_path)
run.finish()
```

`wandb.Run.link_artifact()`를 실행하면 모델 Artifact가 registry 내 `DemoModels` 컬렉션에 추가됩니다. 여기서 Artifact의 버전 히스토리, [계보 맵]({{< relref path="/guides/core/registry/lineage/" lang="ko" >}}), 기타 [메타데이터]({{< relref path="/guides/core/registry/registry_cards/" lang="ko" >}}) 등의 정보를 볼 수 있습니다.

레지스트리에 Artifact를 연결하는 더 자세한 방법은 [Link artifacts to a registry]({{< relref path="/guides/core/registry/link_version/" lang="ko" >}})를 참고하세요.

## 추론을 위해 레지스트리에서 모델 artifact 가져오기

모델을 추론에 사용하려면 `wandb.Run.use_artifact()`를 사용해 registry에서 게시된 Artifact를 가져옵니다. 반환된 artifact 오브젝트에 [`wandb.Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactdownload" lang="ko" >}})를 호출해 Artifact를 로컬 파일로 다운로드할 수 있습니다.

```python
REGISTRY_NAME = "Model"  # W&B 내 registry의 이름
COLLECTION_NAME = "DemoModels"  # 레지스트리 내 컬렉션 이름
VERSION = 0 # 가져올 artifact의 버전

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()
```

레지스트리에서 artifact를 가져오는 더 자세한 방법은 [Download an artifact from a registry]({{< relref path="/guides/core/registry/download_use_artifact/" lang="ko" >}})를 참고하세요.

사용 중인 머신러닝 프레임워크에 따라, 가중치 로드 전에 모델 아키텍처를 재구성해야 할 수도 있습니다. 이 부분은 사용하는 프레임워크와 모델에 따라 달라지므로 독자에게 연습 문제로 남깁니다.

## 발견한 내용을 리포트로 공유하기

{{% alert %}}
W&B Report and Workspace API는 퍼블릭 프리뷰 상태입니다.
{{% /alert %}}

작업 요약을 위해 [리포트]({{< relref path="/guides/core/reports/_index.md" lang="ko" >}})를 생성하고 공유하세요. 프로그래밍적으로 리포트를 생성하려면 [W&B Report and Workspace API]({{< relref path="/ref/python/wandb_workspaces/reports.md" lang="ko" >}})를 사용하세요.

우선, W&B Reports API를 설치합니다:

```python
pip install wandb wandb-workspaces -qqq
```

다음 코드 블록은 마크다운, 패널 그리드 등 여러 블록이 포함된 리포트를 생성합니다. 블록을 추가하거나 내용을 변경해 리포트를 자유롭게 커스터마이즈할 수 있습니다.

코드 실행 결과로 만들어진 리포트의 URL 링크가 출력됩니다. 브라우저에서 해당 링크를 열어 리포트를 확인할 수 있습니다.

```python
import wandb_workspaces.reports.v2 as wr

experiment_summary = """이 실험은 W&B를 사용하여 간단한 모델을 트레이닝한 요약입니다."""
dataset_info = """트레이닝에 사용된 데이터셋은 간단한 모델로 생성된 합성 데이터입니다."""
model_info = """이 모델은 입력 데이터와 일정 노이즈를 이용해 출력을 예측하는 선형 회귀 모델입니다."""

report = wr.Report(
    project=PROJECT,
    entity=TEAM_ENTITY,
    title="My Awesome Model Training Report",
    description=experiment_summary,
    blocks= [
        wr.TableOfContents(),
        wr.H2("Experiment Summary"),
        wr.MarkdownBlock(text=experiment_summary),
        wr.H2("Dataset Information"),
        wr.MarkdownBlock(text=dataset_info),
        wr.H2("Model Information"),
        wr.MarkdownBlock(text = model_info),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(title="Train Loss", x="Step", y=["loss"], title_x="Step", title_y="Loss")
                ],
            ),  
    ]

)

# 리포트를 W&B에 저장
report.save()
```

프로그래밍적으로 리포트를 생성하는 방법, 또는 W&B App에서 인터랙티브하게 리포트를 만드는 방법에 대한 더 자세한 정보는 W&B Docs 개발자 가이드의 [Create a report]({{< relref path="/guides/core/reports/create-a-report.md" lang="ko" >}})를 참고하세요.

## 레지스트리 쿼리하기
[W&B Public APIs]({{< relref path="/ref/python/public-api/_index.md" lang="ko" >}})를 사용해 W&B에 저장된 이력 데이터를 쿼리, 분석, 관리할 수 있습니다. 이 방법은 artifact의 계보 추적, 다양한 버전 비교, 시간이 지남에 따라 모델 성능을 분석하는 데 유용합니다.

아래 코드 블록은 특정 컬렉션 내 모든 Artifact를 쿼리하는 예시입니다. 컬렉션을 가져와 각 버전을 반복하며 Artifact의 이름과 버전을 출력합니다.

```python
import wandb

# wandb API 초기화
api = wandb.Api()

# 문자열 `model`이 포함된 모든 artifact 버전 중 
# `text-classification` 태그가 있거나 `latest` 에일리어스가 있는 artifacts 찾기
registry_filters = {
    "name": {"$regex": "model"}
}

# 논리 $or 연산자로 artifact 버전 필터링
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# 필터에 맞는 모든 artifact 버전을 iterable로 반환
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# artifact의 이름, 소속 컬렉션, 에일리어스, 태그, 생성 일자를 출력
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

레지스트리 쿼리에 대한 더 자세한 정보는 [Query registry items with MongoDB-style queries]({{< relref path="/guides/core/registry/search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ko" >}})를 참고하세요.