---
title: Manage artifact data retention
description: TTL(Time To Live) 정책
menu:
  default:
    identifier: ko-guides-core-artifacts-manage-data-ttl
    parent: manage-data
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" >}}

W&B Artifact time-to-live (TTL) 정책으로 Artifacts가 W&B에서 삭제될 시점을 예약하세요. 아티팩트를 삭제하면 W&B는 해당 아티팩트를 *소프트 삭제*로 표시합니다. 즉, 아티팩트는 삭제될 것으로 표시되지만 파일은 즉시 저장소에서 삭제되지 않습니다. W&B가 Artifacts를 삭제하는 방법에 대한 자세한 내용은 [Artifacts 삭제]({{< relref path="./delete-artifacts.md" lang="ko" >}}) 페이지를 참조하세요.

[이](https://www.youtube.com/watch?v=hQ9J6BoVmnc) 비디오 튜토리얼을 통해 W&B 앱에서 Artifacts TTL로 데이터 보존을 관리하는 방법을 알아보세요.

{{% alert %}}
W&B는 Model Registry에 연결된 모델 Artifacts에 대한 TTL 정책을 설정하는 옵션을 비활성화합니다. 이는 연결된 모델이 프로덕션 워크플로우에서 사용될 때 실수로 만료되지 않도록 하기 위함입니다.
{{% /alert %}}
{{% alert %}}
* 팀 관리자만 [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings.md" lang="ko" >}})을 보고 (1) 누가 TTL 정책을 설정하거나 편집할 수 있는지 허용하거나 (2) 팀 기본 TTL을 설정하는 등 팀 수준 TTL 설정을 엑세스할 수 있습니다.
* W&B 앱 UI에서 Artifacts 세부 정보에 TTL 정책을 설정하거나 편집하는 옵션이 표시되지 않거나 프로그래밍 방식으로 TTL을 설정해도 Artifacts의 TTL 속성이 성공적으로 변경되지 않으면 팀 관리자가 해당 권한을 부여하지 않은 것입니다.
{{% /alert %}}

## 자동 생성된 Artifacts
사용자가 생성한 Artifacts만 TTL 정책을 사용할 수 있습니다. W&B에서 자동으로 생성된 Artifacts는 TTL 정책을 설정할 수 없습니다.

다음 Artifacts 유형은 자동 생성된 Artifacts를 나타냅니다.
- `run_table`
- `code`
- `job`
- `wandb-*`로 시작하는 모든 Artifacts 유형

[W&B 플랫폼]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}}) 또는 프로그래밍 방식으로 Artifacts 유형을 확인할 수 있습니다.

```python
import wandb

run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>`로 묶인 값을 자신의 값으로 바꾸세요.

## TTL 정책을 편집하고 설정할 수 있는 사람 정의
팀 내에서 TTL 정책을 설정하고 편집할 수 있는 사람을 정의합니다. 팀 관리자에게만 TTL 권한을 부여하거나 팀 관리자와 팀 멤버 모두에게 TTL 권한을 부여할 수 있습니다.

{{% alert %}}
팀 관리자만 TTL 정책을 설정하거나 편집할 수 있는 사람을 정의할 수 있습니다.
{{% /alert %}}

1. 팀 프로필 페이지로 이동합니다.
2. **설정** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) 섹션**으로 이동합니다.
4. **TTL 권한 드롭다운**에서 TTL 정책을 설정하고 편집할 수 있는 사람을 선택합니다.
5. **설정 검토 및 저장**을 클릭합니다.
6. 변경 사항을 확인하고 **설정 저장**을 선택합니다.

{{< img src="/images/artifacts/define_who_sets_ttl.gif" alt="" >}}

## TTL 정책 만들기
Artifacts를 만들 때 또는 Artifacts를 만든 후에 소급하여 TTL 정책을 설정합니다.

아래의 모든 코드 조각에서는 코드를 사용하기 위해 `<>`로 묶인 콘텐츠를 자신의 정보로 바꾸세요.

### Artifacts를 만들 때 TTL 정책 설정
W&B Python SDK를 사용하여 Artifacts를 만들 때 TTL 정책을 정의합니다. TTL 정책은 일반적으로 일 단위로 정의됩니다.

{{% alert %}}
Artifacts를 만들 때 TTL 정책을 정의하는 것은 일반적으로 [Artifacts를 만드는 방법]({{< relref path="../construct-an-artifact.md" lang="ko" >}})과 유사합니다. 단, Artifacts의 `ttl` 속성에 시간 델타를 전달한다는 점만 다릅니다.
{{% /alert %}}

단계는 다음과 같습니다.

1. [Artifacts 만들기]({{< relref path="../construct-an-artifact.md" lang="ko" >}}).
2. 파일, 디렉토리 또는 참조와 같은 [Artifacts에 콘텐츠 추가]({{< relref path="../construct-an-artifact.md#add-files-to-an-artifact" lang="ko" >}}).
3. Python 표준 라이브러리의 일부인 [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) 데이터 유형으로 TTL 시간 제한을 정의합니다.
4. [Artifacts 로깅]({{< relref path="../construct-an-artifact.md#3-save-your-artifact-to-the-wb-server" lang="ko" >}}).

다음 코드 조각은 Artifacts를 만들고 TTL 정책을 설정하는 방법을 보여줍니다.

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL 정책 설정
run.log_artifact(artifact)
```

앞의 코드 조각은 Artifacts에 대한 TTL 정책을 30일로 설정합니다. 즉, W&B는 30일 후에 Artifacts를 삭제합니다.

### Artifacts를 만든 후 TTL 정책 설정 또는 편집
W&B 앱 UI 또는 W&B Python SDK를 사용하여 이미 존재하는 Artifacts에 대한 TTL 정책을 정의합니다.

{{% alert %}}
Artifacts의 TTL을 수정하면 Artifacts가 만료되는 시간은 Artifacts의 `createdAt` 타임스탬프를 사용하여 계속 계산됩니다.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [Artifacts 가져오기]({{< relref path="../download-and-use-an-artifact.md" lang="ko" >}}).
2. Artifacts의 `ttl` 속성에 시간 델타를 전달합니다.
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ko" >}}) 메소드로 Artifacts를 업데이트합니다.


다음 코드 조각은 Artifacts에 대한 TTL 정책을 설정하는 방법을 보여줍니다.
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2년 후에 삭제
artifact.save()
```

앞의 코드 예제는 TTL 정책을 2년으로 설정합니다.
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B 앱 UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 선택합니다.
3. Artifacts 목록에서 TTL 정책을 편집할 Artifacts 유형을 확장합니다.
4. TTL 정책을 편집할 Artifacts 버전을 선택합니다.
5. **버전** 탭을 클릭합니다.
6. 드롭다운에서 **TTL 정책 편집**을 선택합니다.
7. 나타나는 모달 내에서 TTL 정책 드롭다운에서 **사용자 지정**을 선택합니다.
8. **TTL 기간** 필드 내에서 TTL 정책을 일 단위로 설정합니다.
9. **TTL 업데이트** 버튼을 선택하여 변경 사항을 저장합니다.

{{< img src="/images/artifacts/edit_ttl_ui.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}



### 팀의 기본 TTL 정책 설정

{{% alert %}}
팀 관리자만 팀의 기본 TTL 정책을 설정할 수 있습니다.
{{% /alert %}}

팀의 기본 TTL 정책을 설정합니다. 기본 TTL 정책은 각 생성 날짜를 기준으로 기존 및 향후 Artifacts에 모두 적용됩니다. 기존 버전 수준 TTL 정책이 있는 Artifacts는 팀의 기본 TTL의 영향을 받지 않습니다.

1. 팀 프로필 페이지로 이동합니다.
2. **설정** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) 섹션**으로 이동합니다.
4. **팀의 기본 TTL 정책 설정**을 클릭합니다.
5. **기간** 필드 내에서 TTL 정책을 일 단위로 설정합니다.
6. **설정 검토 및 저장**을 클릭합니다.
7/ 변경 사항을 확인한 다음 **설정 저장**을 선택합니다.

{{< img src="/images/artifacts/set_default_ttl.gif" alt="" >}}

### run 외부에서 TTL 정책 설정

공용 API를 사용하여 run을 가져오지 않고 Artifacts를 검색하고 TTL 정책을 설정합니다. TTL 정책은 일반적으로 일 단위로 정의됩니다.

다음 코드 샘플은 공용 API를 사용하여 Artifacts를 가져오고 TTL 정책을 설정하는 방법을 보여줍니다.

```python
api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

artifact.ttl = timedelta(days=365)  # 1년 후에 삭제

artifact.save()
```

## TTL 정책 비활성화
W&B Python SDK 또는 W&B 앱 UI를 사용하여 특정 Artifacts 버전에 대한 TTL 정책을 비활성화합니다.



{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}
1. [Artifacts 가져오기]({{< relref path="../download-and-use-an-artifact.md" lang="ko" >}}).
2. Artifacts의 `ttl` 속성을 `None`으로 설정합니다.
3. [`save`]({{< relref path="/ref/python/run.md#save" lang="ko" >}}) 메소드로 Artifacts를 업데이트합니다.


다음 코드 조각은 Artifacts에 대한 TTL 정책을 해제하는 방법을 보여줍니다.
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```
  {{% /tab %}}
  {{% tab header="W&B App" %}}
1. W&B 앱 UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 Artifacts 아이콘을 선택합니다.
3. Artifacts 목록에서 TTL 정책을 편집할 Artifacts 유형을 확장합니다.
4. TTL 정책을 편집할 Artifacts 버전을 선택합니다.
5. 버전 탭을 클릭합니다.
6. **레지스트리에 연결** 버튼 옆에 있는 미트볼 UI 아이콘을 클릭합니다.
7. 드롭다운에서 **TTL 정책 편집**을 선택합니다.
8. 나타나는 모달 내에서 TTL 정책 드롭다운에서 **비활성화**를 선택합니다.
9. **TTL 업데이트** 버튼을 선택하여 변경 사항을 저장합니다.

{{< img src="/images/artifacts/remove_ttl_polilcy.gif" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}




## TTL 정책 보기
Python SDK 또는 W&B 앱 UI로 Artifacts에 대한 TTL 정책을 봅니다.

{{< tabpane text=true >}}
  {{% tab  header="Python SDK" %}}
인쇄 문을 사용하여 Artifacts의 TTL 정책을 봅니다. 다음 예제는 Artifacts를 검색하고 해당 TTL 정책을 보는 방법을 보여줍니다.

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```
  {{% /tab %}}
  {{% tab  header="W&B App" %}}
W&B 앱 UI로 Artifacts에 대한 TTL 정책을 봅니다.

1. [https://wandb.ai](https://wandb.ai)에서 W&B 앱으로 이동합니다.
2. W&B 프로젝트로 이동합니다.
3. 프로젝트 내에서 왼쪽 사이드바에서 Artifacts 탭을 선택합니다.
4. 컬렉션을 클릭합니다.

컬렉션 보기 내에서 선택한 컬렉션의 모든 Artifacts를 볼 수 있습니다. `Time to Live` 열 내에서 해당 Artifacts에 할당된 TTL 정책이 표시됩니다.

{{< img src="/images/artifacts/ttl_collection_panel_ui.png" alt="" >}}
  {{% /tab %}}
{{< /tabpane >}}
