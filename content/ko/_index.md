---
title: Weights & Biases 문서
---

<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1600px; margin: 0 auto">
{{< banner title="Weights & Biases 문서" background="/images/support/support_banner.png" >}}
필요한 제품의 문서를 선택하세요.
{{< /banner >}}

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

### 애플리케이션에서 AI 모델 활용

[W&B Weave](https://weave-docs.wandb.ai/)를 사용해 코드 내에서 AI 모델을 관리할 수 있습니다. 트레이싱, 결과 평가, 비용 산출, 다양한 대형 언어 모델(LLM)과 설정을 비교하는 호스팅된 추론 서비스 및 플레이그라운드를 제공합니다.

- [소개](https://weave-docs.wandb.ai/)
- [퀵스타트](https://weave-docs.wandb.ai/quickstart)
- [YouTube 데모](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [플레이그라운드 체험](https://weave-docs.wandb.ai/guides/tools/playground/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### AI 모델 개발

[W&B Models]({{< relref path="/guides/" lang="ko" >}})를 사용하면 AI 모델 개발을 효율적으로 관리할 수 있습니다. 트레이닝, 파인튜닝, 리포팅, 하이퍼파라미터 스윕 자동화, 모델 레지스트리를 통한 버전 관리와 재현성 지원 기능을 제공합니다.

- [소개]({{< relref path="/guides/" lang="ko" >}})
- [퀵스타트]({{< relref path="/guides/quickstart/" lang="ko" >}})
- [YouTube 튜토리얼](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [온라인 코스](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/inference/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Inference logo" width="40" height="40"/>
</div>
<h2>W&B Inference</h2>

### 파운데이션 모델 엑세스

[W&B Inference]({{< relref path="/guides/inference/" lang="ko" >}})를 통해 OpenAI 호환 API로 선도적인 오픈소스 파운데이션 모델을 만날 수 있습니다. 다양한 모델, 사용량 추적, Weave와 연동된 트레이싱 및 평가를 지원합니다.

- [소개]({{< relref path="/guides/inference/" lang="ko" >}})
- [사용 가능한 모델]({{< relref path="/guides/inference/models/" lang="ko" >}})
- [API Reference]({{< relref path="/guides/inference/api-reference/" lang="ko" >}})
- [플레이그라운드에서 체험](https://wandb.ai/inference)

</div>{{% /card %}}
{{< /cardpane >}}
</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>Core Components</h2>

W&B 제품군은 AI/ML 엔지니어링을 가능하게 하고 가속화하는 공통 컴포넌트를 공유합니다.

- [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})
- [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})
- [Automations]({{< relref path="/guides/core/automations/" lang="ko" >}})
- [Secrets]({{< relref path="/guides/core/secrets.md" lang="ko" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>플랫폼</h2>

Weights & Biases 플랫폼은 SaaS와 온프레미스 모두에서 제공되며, IAM, 보안, 모니터링, 프라이버시 등 다양한 기능을 갖추고 있습니다.

- [배포 옵션]({{< relref path="/guides/hosting/hosting-options/" lang="ko" >}})
- [ID 및 엑세스 관리(IAM)]({{< relref path="/guides/hosting/iam/" lang="ko" >}})
- [데이터 보안]({{< relref path="/guides/hosting/data-security/" lang="ko" >}})
- [프라이버시 설정]({{< relref path="/guides/hosting/privacy-settings/" lang="ko" >}})
- [모니터링 및 사용량]({{< relref path="/guides/hosting/monitoring-usage/" lang="ko" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/support/'" style="cursor: pointer;padding-left:20px;">

<h2>지원</h2>

Weights & Biases 플랫폼 전반에 대한 도움을 받을 수 있습니다. 자주 묻는 질문, 문제 해결 가이드, 지원팀 연락방법을 확인하세요.

- [지식 기반 문서]({{< relref path="/support/" lang="ko" >}})
- [커뮤니티 포럼](https://wandb.ai/community)
- [Discord 서버](https://discord.com/invite/RgB8CPk2ce)
- [지원팀 문의](https://wandb.ai/site/contact/)

</div>{{% /card %}}
{{< /cardpane >}}
</div>




</div>


<style>
.td-card-group { margin: 0 auto }
p { overflow: hidden; display: block; }
ul { margin-left: 50px; }

/* Make all cards uniform size in 3x2 grid */
.top-row-cards .td-card-group,
.bottom-row-cards .td-card-group {
    max-width: 100%;
    display: flex;
    justify-content: center;
}

.td-card {
    max-width: 480px !important;
    min-width: 480px !important;
    margin: 0.75rem !important;
    flex: 0 0 auto;
}

/* Ensure consistent height for all cards */
.td-card .card {
    height: 100%;
    min-height: 320px;
}
</style>