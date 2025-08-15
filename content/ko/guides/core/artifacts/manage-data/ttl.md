---
title: 아티팩트 데이터 보존 관리
description: 수명 정책(TTL)
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

Artifacts 의 보존 기간( Time-to-live, TTL) 정책을 통해 W&B 에서 아티팩트가 삭제되는 시점을 예약할 수 있습니다. 아티팩트를 삭제하면, W&B 는 해당 아티팩트를 *소프트-삭제*로 표시합니다. 즉, 아티팩트가 삭제 예정으로 표시되지만 파일이 즉시 스토리지에서 삭제되지는 않습니다. W&B가 아티팩트를 어떻게 삭제하는지 더 알고 싶다면 [아티팩트 삭제]({{< relref path="./delete-artifacts.md" lang="ko" >}}) 페이지를 참고하세요.

[Artifacts TTL 을 활용한 데이터 보존 관리](https://www.youtube.com/watch?v=hQ9J6BoVmnc) 비디오 튜토리얼을 시청하여 W&B App 에서 Artifacts TTL로 데이터 보존을 관리하는 방법을 배워보세요.

{{% alert %}}
Model Registry 에 연결된 모델 아티팩트의 경우 TTL 정책 설정 옵션이 비활성화됩니다. 이는 프로덕션 워크플로우에서 사용하는 연결된 모델이 만료되어 삭제되는 상황을 방지하기 위함입니다.
{{% /alert %}}
{{% alert %}}
* 팀 관리자만 [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ko" >}}) 페이지에서 팀 수준의 TTL 설정을 볼 수 있으며, (1) 누가 TTL 정책을 설정/수정할 수 있는지, (2) 팀 기본 TTL 을 설정할 수 있습니다.  
* W&B App UI 에서 아티팩트 상세 정보에서 TTL 정책 설정/수정 옵션이 보이지 않거나, 프로그래밍적으로 TTL 을 설정해도 정상적으로 변경되지 않는 경우, 해당 권한이 팀 관리자에 의해 부여되지 않았기 때문입니다.
{{% /alert %}}

## 자동 생성된 Artifacts
사용자가 직접 생성한 artifact야만 TTL 정책을 사용할 수 있습니다. W&B 가 자동으로 생성한 artifact는 TTL 정책을 적용할 수 없습니다.

자동 생성 Artifacts 타입은 다음과 같습니다:
- `run_table`
- `code`
- `job`
- `wandb-*`로 시작하는 모든 artifact 타입

Artifact 의 타입은 [W&B 플랫폼]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}})에서 직접 확인할 수도 있고, 코드로도 확인할 수 있습니다.

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>`로 감싸진 부분을 여러분의 실제 값으로 바꿔주세요.

## TTL 정책 설정·수정 권한 정의하기
팀 내에서 누가 TTL 정책을 설정하고 수정할 수 있는지 정할 수 있습니다. TTL 권한은 팀 관리자만 가지도록 하거나, 팀 멤버 전체가 가지도록 할 수 있습니다.

{{% alert %}}
TTL 정책을 제한/수정할 수 있는 권한은 팀 관리자만 부여할 수 있습니다.
{{% /alert %}}

1. 팀 프로필 페이지로 이동합니다.
2. **Settings** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) 섹션**으로 이동합니다.
4. **TTL permissions dropdown**에서 TTL 정책을 설정/수정할 수 있는 대상을 선택합니다.  
5. **Review and save settings**를 클릭합니다.
6. 변경 사항을 확인한 후 **Save settings**를 클릭합니다.

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="Setting TTL permissions" >}}

## TTL 정책 생성하기
아티팩트를 생성할 때나 이미 생성된 아티팩트에 대해 TTL 정책을 설정할 수 있습니다.

아래 코드조각에서는 `<>`로 감싼 부분을 여러분의 정보로 교체해서 사용하세요.

### 아티팩트 생성 시 TTL 정책 바로 설정하기
W&B Python SDK 를 활용하여 artifact 생성과 동시에 TTL 정책을 지정할 수 있습니다. TTL 값은 보통 일 단위로 정의합니다.    

{{% alert %}}
아티팩트 생성 시 TTL 정책을 정의하는 것은 일반적인 [artifact 생성]({{< relref path="../construct-an-artifact.md" lang="ko" >}}) 방법과 유사합니다. 단, 시간차를 의미하는 값을 artifact 의 `ttl` 속성에 전달하면 됩니다.
{{% /alert %}}

진행 단계는 아래와 같습니다:

1. [아티팩트 생성]({{< relref path="../construct-an-artifact.md" lang="ko" >}})
2. [파일, 디렉토리, 참조 등 아티팩트에 내용 추가]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ko" >}})
3. Python 표준 라이브러리의 [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) 타입으로 TTL 시간 제한을 정의
4. [아티팩트 로그]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ko" >}})

아래 코드조각은 아티팩트를 생성하고 TTL 정책을 설정하는 방법을 보여줍니다.

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL 정책 설정
run.log_artifact(artifact)
```

위 코드조각은 아티팩트의 TTL 정책을 30일로 설정합니다. 즉, 30일 후 해당 아티팩트는 W&B 에서 삭제됩니다.

### 생성된 아티팩트에 TTL 정책 추가·수정하기
이미 생성된 아티팩트에도 W&B App UI 또는 Python SDK 를 이용해 TTL 정책을 설정할 수 있습니다.

{{% alert %}}
아티팩트의 TTL 을 수정하더라도, expire(만료)까지 남은 시간은 아티팩트가 최초 생성된 시점의 `createdAt` 타임스탬프 기준으로 계산됩니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [아티팩트 가져오기]({{< relref path="../download-and-use-an-artifact.md" lang="ko" >}})
2. artifact 의 `ttl` 속성에 시간차를 전달
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ko" >}}) 메소드로 artifact 업데이트

아래 코드조각은 아티팩트에 TTL 정책을 설정하는 예시입니다:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2년 후 삭제
artifact.save()
```

위 코드는 TTL 정책을 2년(730일)로 설정합니다.
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI 내에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 클릭합니다.
3. 아티팩트 목록에서 원하는 타입을 펼칩니다.
4. TTL 정책을 수정할 아티팩트 버전을 선택합니다.
5. **Version** 탭을 클릭합니다.
6. 드롭다운에서 **Edit TTL policy**를 선택합니다.
7. 뜨는 모달창에서 TTL 정책 드롭다운에서 **Custom**을 선택합니다.
8. **TTL duration** 필드에 일 단위로 TTL 정책을 입력합니다.
9. **Update TTL** 버튼을 눌러 변경사항을 저장합니다.

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="Editing TTL policy" >}}  
  {{% /tab %}}
{{< /tabpane >}}

### 팀의 기본 TTL 정책 설정하기

{{% alert %}}
팀의 기본 TTL 정책은 팀 관리자만 설정할 수 있습니다.
{{% /alert %}}

팀별 기본 TTL 정책을 설정할 수 있습니다. 기본 TTL 정책은 각 아티팩트의 생성일을 기준으로 모든 기존 및 향후 생성되는 artifact에 적용됩니다. 단, 이미 버전별 TTL 정책이 지정된 artifact에는 적용되지 않습니다.

1. 팀 프로필 페이지로 이동합니다.
2. **Settings** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) 섹션**으로 이동합니다.
4. **Set team's default TTL policy**를 클릭합니다.
5. **Duration** 필드에 일 단위로 TTL 정책을 입력합니다.
6. **Review and save settings**를 클릭합니다.
7. 변경 내용을 확인한 후 **Save settings**를 선택합니다.

{{< img src="/images/artifacts/set_default_ttl.gif" alt="Setting default TTL policy" >}}

### run 없이 TTL 정책 설정하기

public API를 사용해 run 을 가져오지 않고, artifact 를 직접 조회해 TTL 정책을 설정할 수 있습니다. TTL 정책은 보통 일 단위로 지정합니다.

아래는 public API로 artifact를 조회하고 TTL 정책을 지정하는 예시입니다.

```python 
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1년 후 삭제

artifact.save()
```

## TTL 정책 비활성화(삭제)
특정 artifact 버전에 대해 TTL 정책을 비활성화하려면 W&B Python SDK 또는 W&B App UI 를 사용할 수 있습니다.

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [아티팩트 가져오기]({{< relref path="../download-and-use-an-artifact.md" lang="ko" >}})
2. artifact 의 `ttl` 값을 `None`으로 설정
3. [`save`]({{< relref path="/ref/python/sdk/classes/run.md#save" lang="ko" >}}) 메소드로 artifact 업데이트

아래 코드는 아티팩트에 TTL 정책을 해제하는 방법을 보여줍니다.
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B App UI 내에서 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 클릭합니다.
3. 아티팩트 목록에서 원하는 타입을 펼칩니다.
4. TTL 정책을 수정할 아티팩트 버전을 선택합니다.
5. Version 탭을 클릭합니다.
6. **Link to registry** 버튼 옆의 meatball(점3개) 아이콘 클릭
7. 드롭다운에서 **Edit TTL policy** 선택
8. 뜨는 모달창에서 TTL 정책 드롭다운에서 **Deactivate** 선택
9. **Update TTL** 버튼을 눌러 변경사항 저장

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="Removing TTL policy" >}}  
  {{% /tab %}}
{{< /tabpane >}}

## TTL 정책 확인하기
Python SDK 또는 W&B App UI 를 통해 artifact의 TTL 정책을 확인할 수 있습니다.

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
print 문을 사용해서 artifact의 TTL 정책을 확인할 수 있습니다. 예시:

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```  
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B App UI를 통해 artifact의 TTL 정책을 확인할 수 있습니다.

1. [W&B App](https://wandb.ai) 으로 이동합니다.
2. W&B Project 에 접속합니다.
3. 프로젝트 안에서 왼쪽 사이드바의 Artifacts 탭을 선택합니다.
4. 원하는 컬렉션을 클릭합니다.

컬렉션 뷰에서는 선택한 컬렉션 내의 모든 artifact 가 보이며, `Time to Live` 컬럼에서 artifact에 지정된 TTL 정책을 확인할 수 있습니다.

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="TTL collection view" >}}  
  {{% /tab %}}
{{< /tabpane >}}