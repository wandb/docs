---
title: Explore artifact graphs
description: 자동으로 생성된 직접 비순환 W&B Artifact 그래프를 트래버스합니다.
menu:
  default:
    identifier: ko-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B는 주어진 run이 기록한 Artifacts와 주어진 run이 사용하는 Artifacts를 자동으로 추적합니다. 이러한 Artifacts에는 데이터셋, 모델, 평가 결과 등이 포함될 수 있습니다. Artifact의 계보를 탐색하여 기계 학습 라이프사이클 전반에 걸쳐 생성된 다양한 Artifacts를 추적하고 관리할 수 있습니다.

## 계보
Artifact의 계보를 추적하면 다음과 같은 주요 이점이 있습니다.

- 재현성: 모든 Artifacts의 계보를 추적함으로써 팀은 실험, 모델 및 결과를 재현할 수 있습니다. 이는 디버깅, 실험 및 기계 학습 모델 검증에 필수적입니다.

- 버전 관리: Artifact 계보는 Artifacts의 버전 관리와 시간 경과에 따른 변경 사항 추적을 포함합니다. 이를 통해 팀은 필요한 경우 이전 버전의 데이터 또는 모델로 롤백할 수 있습니다.

- 감사: Artifacts 및 해당 변환에 대한 자세한 기록을 통해 조직은 규제 및 거버넌스 요구 사항을 준수할 수 있습니다.

- 협업 및 지식 공유: Artifact 계보는 시도에 대한 명확한 기록과 무엇이 작동했고 무엇이 작동하지 않았는지 제공함으로써 팀 멤버 간의 더 나은 협업을 촉진합니다. 이는 노력의 중복을 피하고 개발 프로세스를 가속화하는 데 도움이 됩니다.

### Artifact의 계보 찾기
**Artifacts** 탭에서 Artifact를 선택하면 해당 Artifact의 계보를 볼 수 있습니다. 이 그래프 보기는 파이프라인의 일반적인 개요를 보여줍니다.

Artifact 그래프를 보려면:

1. W&B App UI에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifact 아이콘을 선택합니다.
3. **Lineage**를 선택합니다.

{{< img src="/images/artifacts/lineage1.gif" alt="Getting to the Lineage tab" >}}

### 계보 그래프 탐색

제공하는 Artifact 또는 job 유형은 이름 앞에 표시되며, Artifacts는 파란색 아이콘으로, runs는 녹색 아이콘으로 표시됩니다. 화살표는 그래프에서 run 또는 Artifact의 입력 및 출력을 자세히 설명합니다.

{{< img src="/images/artifacts/lineage2.png" alt="Run and artifact nodes" >}}

{{% alert %}}
왼쪽 사이드바와 **Lineage** 탭에서 Artifact의 유형과 이름을 모두 볼 수 있습니다.
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="Inputs and outputs" >}}

더 자세한 보기를 보려면 개별 Artifact 또는 run을 클릭하여 특정 오브젝트에 대한 자세한 정보를 얻으십시오.

{{< img src="/images/artifacts/lineage3a.gif" alt="Previewing a run" >}}

### Artifact 클러스터

그래프 수준에 5개 이상의 runs 또는 Artifacts가 있는 경우 클러스터가 생성됩니다. 클러스터에는 특정 버전의 runs 또는 Artifacts를 찾기 위한 검색 창이 있으며 클러스터 내부의 노드 계보를 계속 조사하기 위해 클러스터에서 개별 노드를 가져옵니다.

노드를 클릭하면 노드에 대한 개요가 있는 미리보기가 열립니다. 화살표를 클릭하면 개별 run 또는 Artifact가 추출되어 추출된 노드의 계보를 검사할 수 있습니다.

{{< img src="/images/artifacts/lineage3b.gif" alt="Searching a run cluster" >}}

## API를 사용하여 계보 추적
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ko" >}})를 사용하여 그래프를 탐색할 수도 있습니다.

Artifact를 만듭니다. 먼저 `wandb.init`으로 run을 만듭니다. 그런 다음 `wandb.Artifact`로 새 Artifact를 만들거나 기존 Artifact를 검색합니다. 다음으로 `.add_file`로 Artifact에 파일을 추가합니다. 마지막으로 `.log_artifact`로 Artifact를 run에 기록합니다. 완성된 코드는 다음과 같습니다.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

Artifact 오브젝트의 [`logged_by`]({{< relref path="/ref/python/artifact.md#logged_by" lang="ko" >}}) 및 [`used_by`]({{< relref path="/ref/python/artifact.md#used_by" lang="ko" >}}) 메서드를 사용하여 Artifact에서 그래프를 탐색합니다.

```python
# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 다음 단계
- [Artifacts를 더 자세히 살펴보기]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ko" >}})
- [Artifact 스토리지 관리]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ko" >}})
- [Artifacts 프로젝트 살펴보기](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)
