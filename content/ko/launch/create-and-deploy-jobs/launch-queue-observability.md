---
title: Monitor launch queue
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-launch-queue-observability
    parent: create-and-deploy-jobs
url: guides/launch/launch-queue-observability
---

대화형 **Queue monitoring dashboard** 를 사용하여 Launch 대기열의 사용량이 많은지 유휴 상태인지 확인하고, 실행 중인 워크로드를 시각화하고, 비효율적인 작업을 찾아보세요. Launch 대기열 대시보드는 컴퓨팅 하드웨어나 클라우드 리소스를 효과적으로 사용하고 있는지 여부를 결정하는 데 특히 유용합니다.

더 자세한 분석을 위해 페이지는 W&B experiment 추적 워크스페이스와 Datadog, NVIDIA Base Command 또는 클라우드 콘솔과 같은 외부 인프라 모니터링 제공업체에 대한 링크를 제공합니다.

{{% alert %}}
Queue monitoring dashboard는 현재 W&B Multi-tenant Cloud 배포 옵션에서만 사용할 수 있습니다.
{{% /alert %}}

## 대시보드 및 플롯
**Monitor** 탭을 사용하여 지난 7일 동안 발생한 대기열의 활동을 확인하세요. 왼쪽 패널을 사용하여 시간 범위, 그룹화 및 필터를 제어합니다.

대시보드에는 성능 및 효율성에 대한 자주 묻는 질문에 답변하는 다양한 플롯이 포함되어 있습니다. 다음 섹션에서는 대기열 대시보드의 UI 요소를 설명합니다.

### 작업 상태
**작업 상태** 플롯은 각 시간 간격으로 실행 중, 대기 중, 큐에 대기 중 또는 완료된 작업 수를 보여줍니다. **작업 상태** 플롯을 사용하여 대기열의 유휴 기간을 식별합니다.

{{< img src="/images/launch/launch_obs_jobstatus.png" alt="" >}}

예를 들어 고정 리소스 (예: DGX BasePod)가 있다고 가정합니다. 고정 리소스로 유휴 대기열을 관찰하는 경우 스윕과 같은 우선 순위가 낮은 선점형 Launch 작업을 실행할 수 있는 기회를 제시할 수 있습니다.

반면에 클라우드 리소스를 사용하고 주기적인 활동 버스트가 표시된다고 가정합니다. 주기적인 활동 버스트는 특정 시간에 리소스를 예약하여 비용을 절약할 수 있는 기회를 제시할 수 있습니다.

플롯의 오른쪽에는 [Launch 작업 상태]({{< relref path="./launch-view-jobs.md#check-the-status-of-a-job" lang="ko" >}})를 나타내는 색상을 보여주는 키가 있습니다.

{{% alert %}}
`Queued` 항목은 워크로드를 다른 대기열로 이동할 수 있는 기회를 나타낼 수 있습니다. 실패 급증은 Launch 작업 설정에 도움이 필요한 사용자를 식별할 수 있습니다.
{{% /alert %}}

### 대기 시간

**대기 시간** 플롯은 지정된 날짜 또는 시간 범위에 대해 Launch 작업이 대기열에 있었던 시간 (초)을 보여줍니다.

{{< img src="/images/launch/launch_obs_queuedtime.png" alt="" >}}

x축은 사용자가 지정하는 시간 프레임을 보여주고 y축은 Launch 작업이 Launch 대기열에 있었던 시간 (초)을 보여줍니다. 예를 들어 특정 날짜에 10개의 Launch 작업이 큐에 있다고 가정합니다. 해당 10개의 Launch 작업이 평균 60초씩 기다리면 **대기 시간** 플롯은 600초를 보여줍니다.

{{% alert %}}
**대기 시간** 플롯을 사용하여 긴 대기열 시간의 영향을 받는 사용자를 식별합니다.
{{% /alert %}}

왼쪽 막대의 `Grouping` 컨트롤을 사용하여 각 작업의 색상을 사용자 정의합니다.

이는 어떤 사용자와 작업이 부족한 대기열 용량으로 인해 어려움을 겪고 있는지 식별하는 데 특히 유용할 수 있습니다.

### 작업 Runs

{{< img src="/images/launch/launch_obs_jobruns2.png" alt="" >}}

이 플롯은 시간 간격으로 실행된 모든 작업의 시작과 끝을 보여주며 각 run에 대해 서로 다른 색상을 사용합니다. 이를 통해 특정 시간에 대기열에서 어떤 워크로드를 처리하고 있는지 한눈에 쉽게 알 수 있습니다.

패널 오른쪽 하단의 Select 툴을 사용하여 작업을 브러시하여 아래 표에 세부 정보를 채웁니다.

### CPU 및 GPU 사용량
**GPU use by a job**, **CPU use by a job**, **GPU memory by job** 및 **System memory by job**을 사용하여 Launch 작업의 효율성을 확인합니다.

{{< img src="/images/launch/launch_obs_gpu.png" alt="" >}}

예를 들어 **GPU memory by job**을 사용하여 W&B run을 완료하는 데 시간이 오래 걸렸는지 여부와 CPU 코어의 낮은 비율을 사용했는지 여부를 확인할 수 있습니다.

각 플롯의 x축은 Launch 작업으로 생성된 W&B run의 지속 시간 (초)을 보여줍니다. 마우스를 데이터 포인트 위에 올려 놓으면 run ID, run이 속한 프로젝트, W&B run을 생성한 Launch 작업 등과 같은 W&B run에 대한 정보를 볼 수 있습니다.

### 오류

**Errors** 패널은 지정된 Launch 대기열에서 발생한 오류를 보여줍니다. 보다 구체적으로 Errors 패널은 오류가 발생한 타임스탬프, 오류가 발생한 Launch 작업의 이름, 생성된 오류 메시지를 보여줍니다. 기본적으로 오류는 최신순에서 가장 오래된 순으로 정렬됩니다.

{{< img src="/images/launch/launch_obs_errors.png" alt="" >}}

**Errors** 패널을 사용하여 사용자를 식별하고 차단을 해제합니다.

## 외부 링크

대기열 관찰 가능성 대시보드의 보기는 모든 대기열 유형에서 일관되지만, 많은 경우 환경별 모니터로 직접 이동하는 것이 유용할 수 있습니다. 이를 위해 대기열 관찰 가능성 대시보드에서 직접 콘솔에서 링크를 추가합니다.

페이지 하단에서 `Manage Links`를 클릭하여 패널을 엽니다. 원하는 페이지의 전체 URL을 추가합니다. 다음으로 레이블을 추가합니다. 추가한 링크는 **External Links** 섹션에 나타납니다.
