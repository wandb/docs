---
title: Artifacts 그래프 탐색하기
description: W&B Artifact 에서 자동으로 생성된 방향성 비순환 그래프를 탐색합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B는 특정 run이 로깅한 Artifacts와 해당 run이 사용하는 Artifacts를 자동으로 추적합니다. 이러한 Artifacts에는 데이터셋, 모델, 평가 결과 등이 포함될 수 있습니다. Artifacts의 계보를 탐색하여 머신러닝 라이프사이클 전반에 걸쳐 생성된 다양한 Artifacts를 추적하고 관리할 수 있습니다.

## 계보 (Lineage)
Artifact의 계보를 추적하면 다음과 같은 주요 이점이 있습니다:

- 재현성: 모든 Artifact의 계보를 추적함으로써 팀은 실험, 모델, 결과를 쉽게 재현할 수 있습니다. 이는 디버깅, 실험, 모델 검증에 매우 유용합니다.

- 버전 관리: Artifact 계보는 Artifacts의 버전과 변화 이력을 관리합니다. 이 덕분에 데이터 또는 모델의 이전 버전으로 쉽게 롤백할 수 있습니다.

- 감사: Artifacts와 그 변환 내역이 상세히 기록되므로, 조직의 규제 또는 거버넌스 준수에 도움이 됩니다.

- 협업 및 지식 공유: 어떤 시도가 있었는지, 성공·실패 사례를 남기므로 팀 내 협업이 쉬워지고, 중복 작업을 줄이며 개발 속도를 높일 수 있습니다.

### Artifact 계보 확인하기
**Artifacts** 탭에서 원하는 Artifact를 선택하면 해당 Artifact의 계보를 볼 수 있습니다. 이 그래프 뷰에서는 전체 파이프라인의 흐름을 한눈에 파악할 수 있습니다.

Artifact 그래프를 보려면 다음처럼 하세요:

1. W&B App UI에서 내 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 클릭합니다.
3. **Lineage**를 선택합니다.

{{< img src="/images/artifacts/lineage1.gif" alt="Getting to the Lineage tab" >}}

### 계보 그래프 탐색

제공된 artifact 또는 job 유형이 Artifact 이름 앞에 표시되며, Artifacts는 파란색 아이콘, runs는 초록색 아이콘으로 나타납니다. 화살표는 그래프에서 run이나 Artifact의 입력 및 출력을 보여줍니다.

{{< img src="/images/artifacts/lineage2.png" alt="Run and artifact nodes" >}}

{{% alert %}}
왼쪽 사이드바와 **Lineage** 탭 어디에서든 Artifact의 타입과 이름을 확인할 수 있습니다.
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="Inputs and outputs" >}}

자세히 살펴보고 싶은 Artifact나 run을 클릭하면 해당 객체의 상세 정보를 확인할 수 있습니다.

{{< img src="/images/artifacts/lineage3a.gif" alt="Previewing a run" >}}

### Artifact 클러스터

그래프 한 레벨에 run이나 Artifact가 5개 이상 있으면 클러스터가 생성됩니다. 클러스터 내에는 특정 run 또는 Artifact 버전을 손쉽게 찾을 수 있도록 검색창이 제공되며, 클러스터 안의 개별 node를 꺼내 그 node의 계보를 이어서 추적할 수 있습니다.

node를 클릭하면 미리보기 창이 열리고, 화살표를 클릭하면 해당 run 또는 Artifact만 남기고 그 계보를 확인할 수 있습니다.

{{< img src="/images/artifacts/lineage3b.gif" alt="Searching a run cluster" >}}

## API로 계보 추적하기
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용해 계보 그래프를 코드로 탐색할 수도 있습니다.

Artifact를 생성하는 방법은 다음과 같습니다. 먼저 `wandb.init`으로 run을 시작합니다. 그리고 `wandb.Artifact`로 새 Artifact를 생성하거나 기존 Artifact를 불러옵니다. `.add_file` 등으로 파일을 추가한 다음, `.log_artifact`로 Artifact를 run에 로깅합니다. 대략적인 코드는 다음과 같습니다:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # 파일 및 데이터를 artifact에 추가합니다.
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` 사용
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

artifact 오브젝트의 [`logged_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#logged_by" lang="ko" >}})와 [`used_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#used_by" lang="ko" >}}) 메서드를 사용해 Artifact와 연결된 run들을 추적할 수 있습니다:

```python
# Artifact에서 시작하여 그래프 위/아래로 탐색합니다.
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 다음 단계
- [Artifacts 더 자세히 살펴보기]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ko" >}})
- [Artifact 스토리지 관리하기]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ko" >}})
- [Artifacts 프로젝트 예시 살펴보기](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)
