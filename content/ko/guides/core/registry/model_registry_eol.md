---
title: 레거시 모델 레지스트리에서 마이그레이션
menu:
  default:
    identifier: ko-guides-core-registry-model_registry_eol
    parent: registry
weight: 9
---

W&B는 기존 **Model Registry**에서 업그레이드된 **W&B Registry**로 이전 중입니다. 이번 전환은 완전히 W&B에 의해 관리되어 사용자가 별다른 작업 없이 원활하게 이루어지도록 설계되었습니다. 마이그레이션 과정에서도 기존 워크플로우는 그대로 유지되며, 강력한 신규 기능을 사용할 수 있게 됩니다. 궁금한 사항이나 지원이 필요하실 경우 [support@wandb.com](mailto:support@wandb.com) 으로 문의해 주세요.

## 마이그레이션 이유

W&B Registry는 기존 Model Registry 대비 다음과 같은 큰 개선사항을 제공합니다:

- **통합된 조직 단위 경험**: 팀과 상관 없이 조직 전체에서 선별된 artifacts를 공유·관리할 수 있습니다.
- **향상된 거버넌스**: 엑세스 제어, 제한된 레지스트리, 가시성 설정 등으로 사용자 엑세스를 세밀하게 관리할 수 있습니다.
- **강화된 기능성**: 커스텀 레지스트리, 더 강력한 검색, 감사 추적, 자동화 지원 등 다양한 신규 기능으로 ML 인프라를 현대화할 수 있습니다.



아래 표는 기존 Model Registry와 새로운 W&B Registry의 주요 차이점을 요약한 것입니다:

| 기능 | 기존 W&B Model Registry | W&B Registry |
| ----- | ----- | ----- |
| 아티팩트 가시성 | 팀 단위만 지원 — 팀 멤버에게만 엑세스 제한 | 조직 단위 가시성과 세분화된 권한 제어 |
| 커스텀 레지스트리 | 미지원 | 완벽 지원 — 모든 아티팩트 유형에 대한 레지스트리 생성 가능 |
| 엑세스 제어 | 미제공 | 레지스트리 단위 역할 기반 엑세스(관리자, 멤버, 뷰어) 지원 |
| 용어 | "Registered models": 모델 버전에 대한 포인터 | "Collections": 모든 아티팩트 버전 포인터 |
| 레지스트리 범위 | 모델 버전 관리만 지원 | 모델, 데이터셋, 커스텀 artifacts 등 다양하게 지원 |
| 자동화 | 레지스트리 단위 자동화 | 레지스트리 및 컬렉션 단위 자동화 모두 지원 및 마이그레이션 시 복사 |
| 검색 및 탐색 | 제한적인 검색·탐색 제공 | 조직 내 모든 레지스트리에 걸쳐 중앙 검색 가능 |
| API 호환성 | `wandb.init.link_model()` 및 MR 전용 패턴 사용 | 최신 SDK API (`link_artifact()`, `use_artifact()`) 자동 리디렉션 포함 |
| 마이그레이션 | 지원 종료 | 자동 마이그레이션 및 기능 향상 — 데이터 복사, 삭제 없음 |

## 마이그레이션 준비

- **따로 조치 필요 없음**: 마이그레이션은 완전히 자동화되어 W&B에서 관리하므로, 스크립트 실행·설정 변경·데이터 수동 이동이 불필요합니다.
- **공지사항 확인**: 마이그레이션 예정일 최소 2주 전에 W&B App UI 내 배너 등으로 안내가 제공됩니다.
- **권한 검토**: 마이그레이션 후에는 관리자(admin)가 레지스트리 엑세스를 점검하여 팀에 적합한 권한 체계를 유지해 주세요.
- **새로운 경로 사용 권장**: 기존 코드는 계속 동작하지만, 신규 프로젝트에는 새로운 W&B Registry 경로 사용을 권장합니다.


## 마이그레이션 절차

### 일시적 쓰기 중단 안내
마이그레이션 과정 중, 데이터 일관성을 위해 팀의 Model Registry에 대한 쓰기 작업이 최대 1시간 동안 일시 중지됩니다. 새로 생성되는 마이그레이션된 W&B Registry에도 같은 기간 쓰기 작업이 일시 중단됩니다.

### 데이터 마이그레이션
W&B는 기존 Model Registry의 다음 데이터를 새 W&B Registry로 이전합니다:

- Collections
- 연결된 아티팩트 버전
- 버전 기록
- 에일리어스, 태그, 설명
- 자동화(컬렉션 및 레지스트리 단위 모두)
- 권한(서비스 계정 역할 및 보호된 에일리어스 등)

W&B App UI 내에서는 기존 Model Registry가 새 W&B Registry로 교체됩니다. 마이그레이션된 레지스트리는 팀 이름 뒤에 `mr-migrated`가 붙은 형태로 표시됩니다:

```text
<team-name>-mr-migrated
```

이 레지스트리들은 기본적으로 **Restricted(제한적)** 가시성으로 제공되어 기존 프라이버시 경계가 유지됩니다. `<team-name>`의 기존 멤버만 해당 레지스트리에 엑세스할 수 있습니다. 


## 마이그레이션 이후

마이그레이션이 완료된 후:

- 기존 Model Registry는 **읽기 전용(read-only)** 이 됩니다. 데이터 조회·엑세스는 가능하나, 신규 쓰기는 불가능합니다.
- 기존 Model Registry의 데이터는 새 W&B Registry로 **복사**됩니다(이동 아님). 삭제되는 데이터는 없습니다.
- 모든 데이터는 새 W&B Registry에서 엑세스할 수 있습니다.
- 버전 관리, 거버넌스, 감사 추적, 자동화 등은 새로운 Registry UI를 통해 사용할 수 있습니다.
- 기존 코드는 계속 동작합니다.
   - [기존 경로 및 API 호출이 자동으로 새로운 W&B Registry로 리디렉션됩니다.]({{< relref path="#code-will-continue-to-work" lang="ko" >}})
   - [아티팩트 버전 경로도 리디렉션됩니다.]({{< relref path="#legacy-paths-will-redirect-to-new-wb-registry-paths" lang="ko" >}})
- 기존 Model Registry는 일정 기간 UI에 계속 노출되며, W&B가 추후 히든 처리할 예정입니다.
- Registry의 향상된 기능을 활용해 보세요:
    - [조직 단위 엑세스]({{< relref path="/guides/core/registry/create_registry/#visibility-types" lang="ko" >}})
    - [역할 기반 엑세스 제어]({{< relref path="/guides/core/registry/configure_registry/" lang="ko" >}})
    - [레지스트리 단위 계보 추적]({{< relref path="/guides/core/registry/lineage/" lang="ko" >}})
    - [자동화]({{< relref path="/guides/core/automations/" lang="ko" >}})

### 기존 코드 호환

기존 코드에서 legacy Model Registry를 참조하는 API 호출은 자동으로 새 W&B Registry로 리디렉션됩니다. 아래와 같은 API 호출들은 별도의 수정 없이 계속 동작합니다:

- `wandb.Api().artifact()`
- `wandb.run.use_artifact()`
- `wandb.run.link_artifact()`
- `wandb.Artifact().link()`

### 기존 경로에서 새 W&B Registry 경로로 리디렉션

W&B는 기존 Model Registry 경로를 자동으로 새 W&B Registry 형식으로 리디렉션합니다. 즉, 기존 코드를 즉시 수정할 필요 없이 사용할 수 있습니다. 단, 자동 리디렉션은 마이그레이션 이전에 legacy Model Registry에 존재하던 컬렉션에 한해 적용됩니다.

예시:
- 기존 Model Registry에 `"my-model"` 컬렉션이 존재했다면, link 동작이 성공적으로 리디렉션됩니다.
- 기존 Model Registry에 `"my-model"` 컬렉션이 없었다면, 리디렉션되지 않고 에러가 발생합니다.

```python
# "my-model"이 legacy Model Registry에 존재했다면 리디렉션 성공
run.link_artifact(artifact, "team-name/model-registry/my-model")

# "new-model"이 legacy Model Registry에 없었다면 실패
run.link_artifact(artifact, "team-name/model-registry/new-model")
```

기존 Model Registry에서 버전 정보 조회를 위해 다음 형식의 경로를 사용했습니다(팀 이름, `"model-registry"`, 컬렉션 이름, 버전으로 구성):

```python
f"{team-name}/model-registry/{collection-name}:{version}"
```

W&B는 이러한 경로도 자동으로 새 W&B Registry 형식(조직 이름, `"wandb-registry"`, 팀 이름, 컬렉션 이름, 버전 포함)으로 리디렉션합니다:

```python
# 새 경로로 리디렉션
f"{org-name}/wandb-registry-{team-name}/{collection-name}:{version}"
```


{{% alert title="Python SDK 경고 안내" %}}

기존 Model Registry 경로를 계속 사용할 경우 경고 메시지가 나타날 수 있습니다. 이 경고는 코드 실행을 중단시키지 않지만, 앞으로는 새 W&B Registry 경로 사용을 권장함을 알립니다.

경고 노출 여부는 W&B Python SDK 버전에 따라 다릅니다:

* 최신 W&B SDK(`v0.21.0` 이상)에서는 리디렉션이 발생했음을 알리는 비중단성 경고가 로그에 표시됩니다.
* 구버전 SDK에서는 경고 없이 조용히 리디렉션됩니다. entity나 project 이름 등 일부 메타데이터가 legacy 값을 그대로 반영할 수 있습니다.

{{% /alert %}} 


## 자주 묻는 질문

### 우리 조직이 언제 마이그레이션되는지 어떻게 알 수 있나요?

W&B에서 UI 내 배너 또는 직접적인 커뮤니케이션을 통해 사전 안내를 드립니다.

### 다운타임(중단 시간)이 있나요?

마이그레이션 중 약 1시간 동안 legacy Model Registry와 새 W&B Registry에 대한 쓰기 작업만 일시 중단됩니다. 그 외 W&B 서비스는 정상적으로 사용 가능합니다.

### 기존 코드가 깨지나요?

아니요. 모든 legacy Model Registry 경로 및 Python SDK 호출은 자동으로 새 Registry로 리디렉션됩니다.

### 데이터가 삭제되나요?

아니요. 모든 데이터는 새 W&B Registry로 복사됩니다. 기존 Model Registry는 읽기 전용으로 한동안 유지되며 이후 숨김 처리됩니다. 데이터는 삭제·분실되지 않습니다.

### 구버전 SDK를 사용하는데 괜찮은가요?

리디렉션은 동일하게 동작하지만, 경고 메시지는 표시되지 않을 수 있습니다. 원활한 경험을 위해 최신 W&B SDK로 업그레이드하시길 권장합니다.

### 마이그레이션된 레지스트리를 이름 변경/수정할 수 있나요?

네, 마이그레이션된 레지스트리에 대해 이름 변경, 멤버 추가·삭제 등 다양한 작업이 가능합니다. 마이그레이션된 레지스트리는 기본적으로 커스텀 레지스트리로 취급되며, 마이그레이션 후에도 리디렉션은 정상 작동합니다.

## 문의

마이그레이션 관련 문의나 지원이 필요하시면 [support@wandb.com](mailto:support@wandb.com) 으로 연락해 주세요. W&B는 새 W&B Registry로의 원활한 전환을 적극 지원합니다.