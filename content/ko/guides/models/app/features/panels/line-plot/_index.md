---
title: 선 그래프
description: 메트릭을 시각화하고, 축을 커스터마이즈하며, 하나의 플롯에서 여러 라인을 비교하세요.
cascade:
- url: guides/app/features/panels/line-plot/:filename
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-_index
    parent: panels
url: guides/app/features/panels/line-plot
weight: 10
---

라인 플롯(line plot)은 `wandb.Run.log()`로 메트릭을 시간에 따라 시각화할 때 기본적으로 표시됩니다. 차트 설정을 활용하면 여러 라인을 한 그래프에서 비교하거나, 커스텀 축을 계산하고, 라벨명을 변경할 수 있습니다.

{{< img src="/images/app_ui/line_plot_example.png" alt="Line plot example" >}}

## 라인 플롯 설정 편집하기

이 섹션에서는 개별 라인 플롯 패널, 하나의 섹션 내 모든 라인 플롯, 또는 워크스페이스 내 모든 라인 플롯의 설정을 수정하는 방법을 안내합니다.

{{% alert %}}
커스텀 x축을 사용하려면, y축을 기록하는 것과 동일한 `wandb.Run.log()` 호출에서 x축도 함께 로그해야 합니다.
{{% /alert %}} 

### 개별 라인 플롯
개별 라인 플롯 설정은 섹션이나 워크스페이스의 라인 플롯 기본 설정을 덮어씁니다. 개별 라인 플롯을 커스터마이즈하려면:

1. 패널 위에 마우스를 올리고, 나타나는 톱니바퀴 아이콘을 클릭하세요.
1. 열리는 패널에서 탭을 선택해 [설정]({{< relref path="#line-plot-settings" lang="ko" >}})을 수정하세요.
1. **적용(Apply)** 을 클릭하세요.

#### 라인 플롯 설정
라인 플롯에서 설정할 수 있는 항목은 다음과 같습니다.

**데이터**: 플롯의 데이터 디스플레이에 관한 세부 설정입니다.
* **X축**: X축에 사용할 값을 선택합니다(기본값은 **Step**). **Relative Time**으로 전환하거나, W&B에 로그한 값 기반의 커스텀 축을 선택할 수 있습니다. X축의 스케일과 범위도 지정할 수 있습니다.
  * **Relative Time (Wall)**: 프로세스가 시작된 이후 실제 시간(시계 시간)을 의미합니다. 예를 들어, run을 시작 후 하루 뒤에 재개하고 값을 기록했다면, 그래프상 24시간 후로 표시됩니다.
  * **Relative Time (Process)**: 실행 중인 프로세스 내부의 시간입니다. run을 시작 후 10초 실행했다가 하루 뒤에 재개하면, 그 시점은 10초로 표시됩니다.
  * **Wall Time**: 그래프에 나타난 첫 run이 시작된 후 경과된 분(minute) 단위 시간입니다.
  * **Step**: `wandb.Run.log()`가 호출될 때마다 기본적으로 증가합니다. 이는 모델에서 기록한 트레이닝 step 수를 반영합니다.
* **Y축**: 기록된 값 중에서 하나 또는 여러 개를 y축으로 선택할 수 있습니다. 시간에 따라 변화하는 메트릭이나 하이퍼파라미터 모두 지정 가능하며, Y축의 스케일과 범위도 설정할 수 있습니다.
* **포인트 집계 방식(Point aggregation method)**: **Random sampling**(기본값) 또는 **Full fidelity** 중 선택할 수 있습니다. 자세한 내용은 [Sampling]({{< relref path="sampling.md" lang="ko" >}}) 문서를 참고하세요.
* **스무딩(Smoothing)**: 선 그래프의 스무딩 유형을 변경합니다. 기본값은 **Time weighted EMA**이며, **No smoothing**, **Running average**, **Gaussian** 등도 선택할 수 있습니다.
* **이상치(Outliers)**: 기본 플롯의 min/max 스케일에서 이상치를 제외할지 설정합니다.
* **최대 run/그룹 수(Max number of runs or groups)**: 이 숫자를 늘리면 한 번에 더 많은 라인을 표시할 수 있습니다(기본 최대 10 runs). 10개를 초과하는 경우 상단에 "Showing first 10 runs" 메시지가 나타납니다.
* **차트 타입(Chart type)**: 라인 플롯, 영역 플롯(area plot), 퍼센트 영역 플롯 중 전환할 수 있습니다.

**그룹화(Grouping)**: 플롯 내에서 run을 그룹핑하고 집계하는 방식 지정
* **Group by**: 지정한 컬럼을 기준으로 같은 값의 run들끼리 그룹핑합니다.
* **Agg**: 그래프 라인의 집계값. 평균(mean), 중앙값(median), 최소(min), 최대값(max) 중 선택 가능

**차트(Chart)**: 패널 및 X/Y축 제목, 축 숨기기/보이기, 범례 위치 지정 등

**범례(Legend)**: 패널 범례(legend)를 활성화했을 때 세부 옵션 지정
* **Legend**: 라인 별로 범례에 나타날 필드명을 지정합니다.
* **Legend template**: 완전히 커스터마이즈된 범례 템플릿을 정의할 수 있습니다. 라인 플롯 상단의 템플릿이나 플롯 위에 마우스를 올렸을 때 나타나는 범례에 어떤 내용이 나올지 지정합니다.

**수식(Expressions)**: 패널에 커스텀 수식(계산식) 추가
* **Y축 수식(Y Axis Expressions)**: 산출(Metric)이나 하이퍼파라미터를 활용하여 계산된 라인을 추가할 수 있습니다.
* **X축 수식(X Axis Expressions)**: 사용자 정의 수식으로 X축을 재스케일할 수 있습니다. 대표적으로 `_step`(기본 x축), `${summary:value}`(서머리 값 참조) 등이 있습니다.

### 섹션 내 모든 라인 플롯

하나의 섹션에 속한 모든 라인 플롯의 기본 설정을 수정하려면(워크스페이스 설정을 덮어씀):

1. 섹션의 톱니바퀴 아이콘을 클릭해 설정을 엽니다.
1. 열리는 패널에서 **Data** 또는 **Display preferences** 탭을 선택해 섹션의 기본 설정을 지정합니다. 각 **Data** 항목은 위의 [개별 라인 플롯]({{< relref path="#line-plot-settings" lang="ko" >}}) 섹션을 참고하세요. Display preference에 대한 자세한 내용은 [섹션 레이아웃 구성]({{< relref path="../#configure-section-layout" lang="ko" >}})을 참고하세요.

### 워크스페이스 내 모든 라인 플롯 
워크스페이스 내 모든 라인 플롯의 기본 설정을 수정하려면:

1. 워크스페이스의 **Settings**(톱니바퀴) 메뉴를 클릭합니다.
1. **Line plots** 메뉴를 클릭합니다.
1. 열리는 패널에서 **Data** 또는 **Display preferences** 탭을 선택해 워크스페이스의 기본 설정을 지정합니다.
    - 각 **Data** 항목의 자세한 내용은 위의 [개별 라인 플롯]({{< relref path="#line-plot-settings" lang="ko" >}})을 참조하세요.

    - 각 **Display preferences** 섹션의 내용은 [워크스페이스 표시 환경 설정]({{< relref path="../#configure-workspace-layout" lang="ko" >}})을 참고하세요. 워크스페이스 수준에서는 라인 플롯의 기본 **확대(Zooming)** 동작을 지정할 수 있습니다. 이 옵션은 X축 키가 일치하는 라인 플롯 간의 확대 영역 동기화를 제어하며, 기본적으로 비활성화되어 있습니다.



## 평균값 시각화하기

여러 개의 다른 실험 결과를 평균해서 그래프에 표시하고 싶다면, 테이블의 그룹핑(Grouping) 기능을 사용할 수 있습니다. run 테이블 위의 "Group"을 클릭한 다음, "All"을 선택하면 그래프에 평균값이 표시됩니다.

평균값 적용 전 그래프 예시는 다음과 같습니다:

{{< img src="/images/app_ui/demo_precision_lines.png" alt="Individual precision lines" >}}

다음 이미지는 여러 run을 그룹핑하여 평균값으로 나타낸 그래프입니다.

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Averaged precision lines" >}}

## NaN 값 시각화하기

`wandb.Run.log()`로 NaN 값을 (예: PyTorch tensor 중 NaN 값 포함) 라인 플롯에 표시할 수 있습니다. 예시:

```python
with wandb.init() as run:
    # NaN 값 기록 예시
    run.log({"test": float("nan")})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="NaN value handling" >}}

## 한 차트에서 두 개의 메트릭 비교

{{< img src="/images/app_ui/visualization_add.gif" alt="Adding visualization panels" >}}

1. 페이지 우측 상단의 **Add panels** 버튼을 클릭합니다.
2. 좌측에 생성된 패널에서 Evaluation 드롭다운을 펼칩니다.
3. **Run comparer**를 선택합니다.


## 라인 플롯의 선 색상 변경하기

기본 run 색상이 비교에 적합하지 않은 경우가 있습니다. 이를 개선하기 위해 wandb는 사용자가 직접 색상을 지정할 수 있는 두 가지 방법을 제공합니다.

{{< tabpane text=true >}}
{{% tab header="런 테이블에서 변경" value="run_table" %}}

  각 run은 초기화 시 무작위 색상으로 배정됩니다.

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="Random colors given to runs" >}}

  색상을 클릭하면 컬러 팔레트가 나타나 원하는 색상으로 직접 변경할 수 있습니다.

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="The color palette" >}}

{{% /tab %}}

{{% tab header="차트 범례 설정에서 변경" value="legend_settings" %}}

1. 설정을 바꾸고자 하는 패널 위에 마우스를 올려둡니다.
2. 나타난 연필 아이콘을 클릭합니다.
3. **Legend** 탭을 선택합니다.

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="Line plot legend settings" >}}

{{% /tab %}}
{{< /tabpane >}}

## 서로 다른 x축으로 시각화하기

실험이 수행된 절대 시간이나, 어떤 날짜에 실행됐는지 등으로 보고 싶다면 x축을 변경하면 됩니다. 아래는 step에서 상대 시간, 그리고 wall time으로 전환하는 예시입니다.

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="X-axis time options" >}}

## 영역 플롯

라인 플롯 설정의 고급 탭에서 다양한 플롯 스타일을 선택해 영역 플롯(area plot)이나 퍼센트 영역 플롯을 만들 수 있습니다.

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="Area plot styles" >}}

## 확대/축소 (Zoom)

직사각형 형태로 클릭&드래그하면 x축과 y축을 동시에 확대/축소할 수 있습니다. 이로 인해 x/y축 모두의 영역이 확대됩니다.

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="Plot zoom functionality" >}}

## 차트 범례 숨기기

라인 플롯에서 아래 토글을 이용해 범례를 손쉽게 숨길 수 있습니다.

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="Hide legend toggle" >}}

## run 메트릭 알림 만들기
[Automations]({{< relref path="/guides/core/automations" lang="ko" >}})을 사용해 run 메트릭이 원하는 조건에 도달했을 때 팀에게 알릴 수 있습니다. Automation은 Slack 채널로 메시지를 보내거나 webhook을 실행할 수 있습니다.

라인 플롯에서 해당 지표에 대한 [run 메트릭 알림]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ko" >}})을 빠르게 생성할 수 있습니다:

1. 패널 위에 마우스를 올려 두고, 종(bell) 아이콘을 클릭하세요.
1. 기본 또는 고급 설정 메뉴를 통해 자동화 조건을 지정하세요. 예시로, run 필터를 적용해 자동화 범위를 한정하거나, 절대 임계치를 설정할 수 있습니다.

[Automations]({{< relref path="/guides/core/automations" lang="ko" >}})에 대해 더 알아보세요.

## CoreWeave 인프라 알림 시각화하기

기계학습 실험 중 GPU 오류, 온도 초과 등 인프라 경고를 실시간으로 W&B에 기록하여 모니터링할 수 있습니다. [W&B run]({{< relref path="/guides/models/track/runs/_index" lang="ko" >}}) 중에 [CoreWeave Mission Control](https://www.coreweave.com/mission-control)이 컴퓨트 인프라를 모니터링하게 됩니다.

{{< alert >}}
이 기능은 현재 Preview 상태이며, CoreWeave 클러스터에서 트레이닝할 때만 사용할 수 있습니다. W&B 담당자에게 엑세스 권한을 문의하세요.
{{< /alert >}}

오류 발생 시 CoreWeave에서 그 정보를 W&B로 전달하며, W&B는 해당 정보를 프로젝트 워크스페이스의 run 플롯에 표기합니다. 일부 이슈는 CoreWeave에서 자동으로 해결을 시도하며, 그 결과도 run 페이지에 표출됩니다.

### run에서 인프라 문제 찾기

W&B에서는 SLURM 작업 및 클러스터 노드의 문제 모두를 추적합니다. 인프라 오류를 확인하려면:

1. W&B App에서 내 프로젝트로 이동하세요.
2. **Workspace** 탭에서 프로젝트 워크스페이스를 확인하세요.
3. 인프라 문제를 포함한 run을 검색 및 선택하세요. CoreWeave에서 인프라 이슈를 감지했다면, run의 플롯 위에 느낌표가 표시된 붉은 세로선이 하나 이상 등장합니다.
4. 그래프의 문제 영역을 클릭하거나, 페이지 우측 상단의 **Issues** 버튼을 클릭하세요. CoreWeave가 보고한 문제 리스트가 패널에 나타납니다.

{{< alert title="Tip" >}}
인프라에 문제가 있는 run을 바로 확인하려면, W&B Workspace에서 **Issues** 컬럼을 고정(pinned)해서 이슈가 있는 run만 한눈에 볼 수 있습니다. 컬럼 고정법은 [run 표시 사용자 정의하기]({{< relref path="/guides/models/track/runs/#customize-how-runs-are-displayed" lang="ko" >}})를 참고하세요.
{{< /alert >}}

패널 상단의 **Overall Grafana view**는 SLURM 작업의 Grafana 대시보드로 이동하며, 작업의 시스템 수준 정보를 확인할 수 있습니다. **Issues summary**는 SLURM 작업에서 CoreWeave Mission Control로 전달된 근본 오류 원인을 요약합니다. 문제가 자동으로 해결됐는지 여부 등도 이 섹션에서 확인할 수 있습니다.

{{< img src="/images/app_ui/cw_wb_observability.png" >}}

**All Issues**에는 run 중 발생한 모든 문제가 최근 발생한 순으로 나열됩니다. 작업 이슈와 노드 이슈 경고 모두 리스트에 표시됩니다. 각 알림에는 이슈명, 발생 시각, 해당 이슈의 Grafana 대시보드 링크, 문제 요약 설명이 포함됩니다.

아래 표는 인프라 문제별 경고 예시를 보여줍니다:

| 카테고리 | 경고 예시 |
| -------- | ------------- |
| 노드 가용성 및 준비 상태 | `KubeNodeNotReadyHGX`, `NodeExtendedDownTime` |
| GPU/가속기 오류 | `GPUFallenOffBusHGX`, `GPUFaultHGX`, `NodeTooFewGPUs` |
| 하드웨어 오류 | `HardwareErrorFatal`, `NodeRAIDMemberDegraded` |
| 네트워킹 & DNS | `NodeDNSFailureHGX`, `NodeEthFlappingLegacyNonGPU` |
| 전원, 쿨링, 관리 | `NodeCPUHZThrottle`, `RedfishDown` |
| DPU & NVSwitch | `DPUNcoreVersionBelowDesired`, `NVSwitchFaultHGX` |
| 기타 | `NodePCISpeedRootGBT`, `NodePCIWidthRootSMC` |

오류 유형에 대한 자세한 내용은 [CoreWeave Docs의 SLURM Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts) 문서를 참조하세요.

### 인프라 문제 디버깅

W&B에서 생성한 각 run은 CoreWeave의 단일 SLURM 작업과 매칭됩니다. 실패한 작업의 [Grafana](https://grafana.com/) 대시보드를 보거나 개별 노드의 상태를 확인할 수 있습니다. **Issues** 패널의 **Overview** 섹션에 있는 링크를 통해 SLURM 작업의 Grafana 대시보드로 이동하며, **All Issues** 드롭다운에서 작업 및 노드 이슈, 관련 Grafana 대시보드를 확인할 수 있습니다.

{{< alert title="Note" >}}
Grafana 대시보드는 CoreWeave 계정이 있는 W&B 사용자만 이용할 수 있습니다. W&B와 협의해 조직의 Grafana 연동을 설정하세요.
{{< /alert >}}

이슈에 따라 SLURM 작업 설정을 조정하거나, 해당 노드의 상태를 점검하고, 작업을 재시작하거나 필요에 따라 추가적인 조치를 취해야 할 수 있습니다.

CoreWeave SLURM 작업의 Grafana 연동에 대해 더 알고 싶다면, [CoreWeave Docs의 SLURM/Job Metrics](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts)를 참고하세요. 작업 알림에 대한 자세한 정보는 [Job info: alerts](https://docs.coreweave.com/docs/observability/managed-grafana/sunk/slurm-job-metrics#job-info-alerts#job-info-alerts)에서 확인할 수 있습니다.