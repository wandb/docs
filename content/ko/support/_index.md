---
title: Support
cascade:
- url: support/:filename
menu:
  support:
    identifier: ko-support-_index
    parent: null
no_list: true
type: docs
url: support
---

{{< banner title="무엇을 도와드릴까요?" background="/images/support/support_banner.png" >}}
지원 문서, 제품 설명서,<br>
그리고 W&B 커뮤니티에서 도움을 찾아보세요.
{{< /banner >}}

## 주요 문서

다음은 모든 카테고리에서 가장 자주 묻는 질문입니다.

* [`wandb.init`은 제 트레이닝 프로세스에 어떤 영향을 미치나요?]({{< relref path="./wandbinit_training_process.md" lang="ko" >}})
* [Sweeps에서 사용자 정의 CLI 코맨드를 어떻게 사용하나요?]({{< relref path="./custom_cli_commands_sweeps.md" lang="ko" >}})
* [오프라인에서 메트릭을 저장한 다음 나중에 W&B에 동기화할 수 있나요?]({{< relref path="./same_metric_appearing_more.md" lang="ko" >}})
* [트레이닝 코드에서 run 이름을 어떻게 구성할 수 있나요?]({{< relref path="./configure_name_run_training_code.md" lang="ko" >}})

찾고 있는 내용이 없으면 아래의 [인기 카테고리]({{< relref path="#popular-categories" lang="ko" >}})를 탐색하거나 카테고리를 기반으로 문서를 검색하세요.

## 인기 카테고리

카테고리별로 문서를 찾아보세요.

{{< cardpane >}}
  {{< card >}}
    <a href="index_experiments">
      <h2 className="card-title">Experiments</h2>
      <p className="card-content">기계 학습 Experiments를 추적, 시각화 및 비교합니다.</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_artifacts">
      <h2 className="card-title">Artifacts</h2>
      <p className="card-content">데이터셋, 모델 및 기타 기계 학습 Artifacts의 버전 관리 및 추적</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}
{{< cardpane >}}
  {{< card >}}
    <a href="index_reports">
      <h2 className="card-title">Reports</h2>
      <p className="card-content">작업 내용을 공유하기 위해 상호 작용적인 협업 Reports를 만듭니다.</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_sweeps">
      <h2 className="card-title">Sweeps</h2>
      <p className="card-content">하이퍼파라미터 검색 자동화</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}


{{< card >}}
  <div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
    {{< img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" >}}
  </div>
  <h2>여전히 찾고 있는 것을 찾을 수 없나요?</h2>
  <a href="mailto:support@wandb.com" className="contact-us-button">
    지원팀에 문의하기
  </a>
 {{< /card >}}
