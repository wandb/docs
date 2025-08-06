---
title: 계보 맵 생성 및 보기
description: W&B Registry에서 계보 맵을 생성하세요.
menu:
  default:
    identifier: ko-guides-core-registry-lineage
    parent: registry
weight: 8
---

W&B Registry의 컬렉션 내에서, ML 실험이 사용하는 Artifacts의 기록을 볼 수 있습니다. 이 기록을 _계보 그래프_ 라고 합니다.

{{% pageinfo color="info" %}}
또한, 컬렉션에 속하지 않은 Artifacts도 W&B에 로그하면 계보 그래프로 확인할 수 있습니다.
{{% /pageinfo %}}

계보 그래프에서는 특정 run이 artifact를 로그한 여부를 보여줍니다. 또한 계보 그래프는 어떤 run이 특정 artifact를 입력으로 사용했는지도 나타낼 수 있습니다. 즉, 계보 그래프를 통해 run의 입력과 출력을 모두 확인할 수 있습니다.

예를 들어, 아래 이미지는 ML 실험에서 생성되고 사용된 artifacts를 보여줍니다.

{{< img src="/images/registry/registry_lineage.png" alt="Registry lineage" >}}

왼쪽에서 오른쪽으로 이미지는 다음을 나타냅니다:
1. 여러 runs가 `split_zoo_dataset:v4` artifact를 로그합니다.
2. "rural-feather-20" run이 트레이닝을 위해 `split_zoo_dataset:v4` artifact를 사용합니다.
3. "rural-feather-20" run의 출력은 `zoo-ylbchv20:v0`라는 모델 artifact입니다.
4. "northern-lake-21"이라는 run이 모델 평가를 위해 `zoo-ylbchv20:v0` 모델 artifact를 사용합니다.


## run의 입력 추적하기

artifact를 run의 입력 또는 의존성으로 지정하려면 `wandb.init.use_artifact` API를 사용합니다.

아래 코드조각은 `use_artifact` 사용 방법을 보여줍니다. 꺾쇠 괄호(`< >`)로 표시된 값은 본인의 값으로 변경하세요.

```python
import wandb

# run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# artifact 가져오기 및 의존성으로 지정
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```


## run의 출력 추적하기

([`wandb.init.log_artifact`]({{< relref path="/ref/python/sdk/classes/run.md#log_artifact" lang="ko" >}}))를 사용해 run의 출력으로 artifact를 선언할 수 있습니다.

아래 코드조각은 `wandb.init.log_artifact` API의 사용 예시입니다. 꺾쇠 괄호(`< >`)로 표시된 값은 본인의 값으로 교체하세요.

```python
import wandb

# run 초기화
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# run의 출력으로 artifact 로그
run.log_artifact(artifact_or_path = artifact)
```

Artifacts 생성에 대한 자세한 내용은 [아티팩트 생성하기]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})를 참고하세요.


## 컬렉션에서 계보 그래프 보기

W&B Registry에서 컬렉션에 연결된 artifact의 계보를 확인할 수 있습니다.

1. W&B Registry로 이동합니다.
2. artifact가 포함된 컬렉션을 선택합니다.
3. 드롭다운 메뉴에서 계보 그래프를 보고 싶은 artifact 버전을 클릭합니다.
4. "Lineage" 탭을 선택합니다.


artifact의 계보 그래프 페이지에 들어가면, 해당 그래프에 있는 어떤 노드도 추가 정보를 확인할 수 있습니다.

run 노드를 선택하면, 해당 run의 ID, run 이름, 상태 등 다양한 run 정보를 확인할 수 있습니다. 예시로, 아래 이미지는 `rural-feather-20` run의 정보를 보여줍니다.

{{< img src="/images/registry/lineage_expanded_node.png" alt="Expanded lineage node" >}}

artifact 노드를 선택하면, 해당 artifact의 전체 이름, 타입, 생성 시각, 관련 에일리어스 등 세부 정보를 볼 수 있습니다.

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="Expanded artifact node details" >}}