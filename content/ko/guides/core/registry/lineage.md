---
title: Create and view lineage maps
description: W&B Registry에서 계보 맵을 만드세요.
menu:
  default:
    identifier: ko-guides-core-registry-lineage
    parent: registry
weight: 8
---

W&B 레지스트리의 컬렉션 내에서 ML 실험에서 사용하는 아티팩트의 이력을 볼 수 있습니다. 이 이력을 _계보 그래프_ 라고 합니다.

{{% pageinfo color="info" %}}
컬렉션에 속하지 않은 W&B에 기록하는 아티팩트에 대한 계보 그래프를 볼 수도 있습니다.
{{% /pageinfo %}}

계보 그래프는 아티팩트를 기록하는 특정 run을 보여줄 수 있습니다. 또한 계보 그래프는 어떤 run이 아티팩트를 입력으로 사용했는지도 보여줄 수 있습니다. 다시 말해, 계보 그래프는 run의 입력과 출력을 보여줄 수 있습니다.

예를 들어, 다음 이미지는 ML 실험 전체에서 생성되고 사용된 아티팩트를 보여줍니다.

{{< img src="/images/registry/registry_lineage.png" alt="" >}}

왼쪽에서 오른쪽으로 이미지는 다음을 보여줍니다.
1. 여러 개의 run이 `split_zoo_dataset:v4` 아티팩트를 기록합니다.
2. "rural-feather-20" run은 트레이닝을 위해 `split_zoo_dataset:v4` 아티팩트를 사용합니다.
3. "rural-feather-20" run의 출력은 `zoo-ylbchv20:v0`이라는 모델 아티팩트입니다.
4. "northern-lake-21"이라는 run은 모델을 평가하기 위해 모델 아티팩트 `zoo-ylbchv20:v0`을 사용합니다.

## run의 입력 추적

`wandb.init.use_artifact` API를 사용하여 아티팩트를 run의 입력 또는 종속성으로 표시합니다.

다음 코드 조각은 `use_artifact`를 사용하는 방법을 보여줍니다. 꺾쇠 괄호(`<>`)로 묶인 값을 사용자의 값으로 바꿉니다.

```python
import wandb

# run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 아티팩트를 가져오고 종속성으로 표시
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```

## run의 출력 추적

([`wandb.init.log_artifact`]({{< relref path="/ref/python/run.md#log_artifact" lang="ko" >}}))를 사용하여 아티팩트를 run의 출력으로 선언합니다.

다음 코드 조각은 `wandb.init.log_artifact` API를 사용하는 방법을 보여줍니다. 꺾쇠 괄호(`<>`)로 묶인 값을 사용자의 값으로 바꾸십시오.

```python
import wandb

# run 초기화
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# 아티팩트를 run의 출력으로 기록
run.log_artifact(artifact_or_path = artifact)
```

아티팩트 생성에 대한 자세한 내용은 [아티팩트 생성]({{< relref path="guides/core/artifacts/construct-an-artifact.md" lang="ko" >}})을 참조하십시오.

## 컬렉션에서 계보 그래프 보기

W&B 레지스트리에서 컬렉션에 연결된 아티팩트의 계보를 봅니다.

1. W&B 레지스트리로 이동합니다.
2. 아티팩트가 포함된 컬렉션을 선택합니다.
3. 드롭다운에서 계보 그래프를 보려는 아티팩트 버전을 클릭합니다.
4. "계보" 탭을 선택합니다.

아티팩트의 계보 그래프 페이지에 있으면 해당 계보 그래프의 모든 노드에 대한 추가 정보를 볼 수 있습니다.

run 노드를 선택하여 run의 ID, run의 이름, run의 상태 등과 같은 run의 세부 정보를 봅니다. 예를 들어, 다음 이미지는 `rural-feather-20` run에 대한 정보를 보여줍니다.

{{< img src="/images/registry/lineage_expanded_node.png" alt="" >}}

아티팩트 노드를 선택하여 전체 이름, 유형, 생성 시간 및 관련 에일리어스와 같은 해당 아티팩트의 세부 정보를 봅니다.

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="" >}}
