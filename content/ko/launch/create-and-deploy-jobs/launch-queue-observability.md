---
title: Launch 큐 모니터링
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

인터랙티브 **Queue monitoring dashboard**를 사용하면 launch queue 가 얼마나 많이 사용되고 있는지, 혹은 유휴 상태인지 확인할 수 있고, 실행 중인 워크로드를 시각화하며, 비효율적인 작업(job)을 쉽게 찾아낼 수 있습니다. launch queue dashboard 는 컴퓨팅 하드웨어나 클라우드 리소스를 효과적으로 사용하고 있는지 판단할 때 특히 유용합니다.

더 깊이 있는 분석이 필요한 경우, 해당 페이지에서 W&B experiment tracking workspace 와 Datadog, NVIDIA Base Command, 클라우드 콘솔과 같은 외부 인프라 모니터링 제공업체로 연결할 수 있습니다.

{{% alert %}}
Queue monitoring dashboard 는 현재 W&B Multi-tenant Cloud 배포 옵션에서만 사용 가능합니다.
{{% /alert %}}

## 대시보드 및 플롯
**Monitor** 탭을 사용해 최근 7일 동안 queue 에서 발생한 활동을 볼 수 있습니다. 왼쪽 패널에서는 시간 범위, 그룹화, 필터를 조절할 수 있습니다.

대시보드에는 성능과 효율성에 관한 자주 묻는 질문에 답할 수 있는 여러 플롯이 포함되어 있습니다. 아래 섹션에서는 queue dashboard 의 UI 요소에 대해 설명합니다.

### Job status
**Job status** 플롯에서는 주어진 시간 구간마다 몇 개의 작업이 실행 중인지, 대기 중인지, queue 에 있는지, 완료되었는지 보여줍니다. **Job status** 플롯을 활용해 queue 가 유휴 상태인 시점을 파악할 수 있습니다.

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="Job status timeline" >}}

예를 들어, 고정 리소스(예: DGX BasePod)가 있을 때 queue 가 유휴 상태라면, 우선순위가 낮은 pre-emptible launch 작업(Sweeps 등)을 실행할 기회일 수 있습니다.

반면, 클라우드 리소스에서 주기적으로 작업량이 몰린다면, 특정 시간에만 리소스를 예약해 비용을 절감할 기회일 수 있습니다.

플롯 오른쪽에는 각 색상이 [launch job 의 상태]({{< relref path="./launch-view-jobs.md#check-the-status-of-a-job" lang="ko" >}})를 뜻하는지 안내하는 키가 있습니다.

{{% alert %}}
`Queued` 항목은 작업을 다른 queue 로 옮겨서 병목을 해소할 기회를 가리킬 수 있습니다. 실패 건수가 급증하면 launch job 셋업에 도움이 필요한 사용자를 찾아낼 수 있습니다.
{{% /alert %}}



### Queued time

**Queued time** 플롯은 launch job 이 특정 날짜 또는 시간 범위에서 queue 에 머문 시간(초 단위)을 보여줍니다.

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="Queued time metrics" >}}

x축은 사용자가 지정하는 기간, y축은 launch job 이 launch queue 에 머문 시간(초)을 나타냅니다. 예시로, 하루 동안 10개의 launch job 이 queue 에 대기했고, 각 작업이 평균 60초씩 기다렸다면, **Queue time** 플롯에는 600초가 표시됩니다.

{{% alert %}}
**Queued time** 플롯으로 긴 queue 대기로 영향을 받는 사용자를 파악하세요.
{{% /alert %}}

왼쪽 바의 `Grouping` 컨트롤로 각 job 의 색상을 커스터마이즈할 수 있습니다.

이는 특히 어떤 사용자와 job 이 queue 용량 부족의 영향을 받고 있는지 식별하는 데 유용합니다.

### Job runs

{{< img src="/images/launch/launch_obs_jobruns2.png" alt="Job runs timeline" >}}

이 플롯은 특정 기간 동안 실행된 각 job 의 시작과 종료 시점을 보여주며, 각 run 마다 다른 색으로 구분됩니다. 덕분에 언제 queue 가 어떤 워크로드를 처리하고 있었는지 한눈에 쉽게 파악할 수 있습니다.

패널 하단 오른쪽에 있는 Select 툴을 이용해 작업을 드래그(brush)하면, 아래 표에 상세 정보가 표시됩니다.



### CPU 및 GPU 사용량
**GPU use by a job**, **CPU use by a job**, **GPU memory by job**, **System memory by job** 플롯을 활용해 launch job 의 효율성을 확인할 수 있습니다.

{{< img src="/images/launch/launch_obs_gpu.png" alt="GPU usage metrics" >}}

예를 들어, **GPU memory by job** 을 통해 W&B run 이 실행 완료까지 오랜 시간이 걸렸는지, CPU 코어를 낮은 비율로 사용했는지 쉽게 파악할 수 있습니다.

각 플롯의 x축은 launch job 으로 생성된 W&B run 의 소요 시간(초)을, 데이터 포인트에 마우스를 올리면 run ID, run 이 속한 프로젝트, 해당 W&B run 을 생성한 launch job 등 자세한 정보를 볼 수 있습니다.

### Errors

**Errors** 패널에서는 주어진 launch queue 에서 발생한 에러를 보여줍니다. 구체적으로 에러가 발생한 시각, 에러가 발생한 launch job 의 이름, 에러 메시지가 표시됩니다. 기본적으로 최신 에러가 가장 위에 나옵니다.

{{< img src="/images/launch/launch_obs_errors.png" alt="Error logs panel" >}}

**Errors** 패널을 사용해, 문제를 겪는 사용자를 식별하고 빠르게 문제를 해소할 수 있습니다.

## 외부 링크

queue 관측 대시보드의 뷰는 모든 queue 유형에 대해 동일하지만, 환경별 모니터로 바로 이동하면 더 유용할 때가 많습니다. 이를 위해 queue 관측 대시보드에서 콘솔로 직접 연결되는 링크를 추가할 수 있습니다.

페이지 하단에서 `Manage Links` 버튼을 클릭하면 패널이 열립니다. 원하는 페이지의 전체 URL을 입력하고, 라벨을 추가하세요. 추가한 링크는 **External Links** 섹션에 표시됩니다.