---
title: 포인트 집계
menu:
  default:
    identifier: ko-guides-models-app-features-panels-line-plot-sampling
    parent: line-plot
weight: 20
---

라인 플롯에서 포인트 집계(point aggregation) 방식을 활용하면 데이터 시각화의 정확성과 성능을 개선할 수 있습니다. 포인트 집계 모드는 [full fidelity]({{< relref path="#full-fidelity" lang="ko" >}})와 [random sampling]({{< relref path="#random-sampling" lang="ko" >}}) 두 가지가 있습니다. 기본적으로 W&B는 full fidelity 모드를 사용합니다.

## Full fidelity

Full fidelity 모드에서는, W&B가 x-축을 데이터 포인트 수에 따라 동적으로 여러 버킷으로 나눕니다. 각 버킷 안에서 최소값, 최대값, 평균값을 계산해 라인 플롯에 포인트 집계 결과를 그립니다.

Full fidelity 모드로 포인트 집계를 사용할 때의 주요 장점 세 가지는 다음과 같습니다:

* 극단값, 피크(스파이크) 보존: 데이터의 극단값과 피크를 그대로 유지할 수 있습니다.
* 최소/최대값 시각화 방식 설정: W&B App에서 극단값(최소/최대값)의 영역을 음영 처리로 표시할지 아닌지인터랙티브하게 결정할 수 있습니다.
* 데이터 손실 없이 분석: 특정 데이터 포인트를 확대(zoom)할 때마다, W&B가 x-축 버킷 크기를 재계산하여 데이터의 정확성이 유지됩니다. 이미 계산된 집계 결과는 캐싱되어, 대용량 데이터셋 탐색 시에도 로딩 속도를 줄일 수 있습니다.

### 최소/최대값 시각화 방식 설정

라인 플롯 주위에 음영 처리를 통해 최소값과 최대값을 표시하거나 숨길 수 있습니다.

아래 이미지는 파란색 라인 플롯을 보여줍니다. 연한 파란색 음영 부분은 각 버킷에서 최소값과 최대값 구간을 나타냅니다.

{{< img src="/images/app_ui/shaded-areas.png" alt="Shaded confidence areas" >}}

라인 플롯에서 최소/최대값을 시각화하는 방법은 세 가지가 있습니다:

* **Never**: 최소/최대값이 음영 구간으로 표시되지 않습니다. x-축 버킷을 따라 집계된 선만 표시됩니다.
* **On hover**: 차트 위에 마우스를 올리면 동적으로 최소/최대값의 음영 구간이 표시됩니다. 기본 뷰는 깔끔하게 유지하면서, 범위를 필요할 때 직접 확인할 수 있습니다.
* **Always**: 차트 내 모든 버킷에 대해 항상 최소/최대값 음영 구역이 표시됩니다. 전체 값의 범위를 한눈에 볼 수 있지만, 여러 run이 동시에 그려진 경우 시각적 노이즈가 많아질 수 있습니다.

기본적으로 최소/최대값 음영 구역은 표시되지 않습니다. 음영 처리 옵션을 적용하려면, 아래 단계를 따라주세요:

{{< tabpane text=true >}}
{{% tab header="워크스페이스 내 모든 차트" value="all_charts" %}}
1. 원하는 W&B 프로젝트로 이동하세요.
2. 왼쪽 탭에서 **Workspace** 아이콘을 클릭합니다.
3. 화면 우측 상단의 **Add panels** 버튼 왼쪽에 위치한 톱니바퀴(설정) 아이콘을 클릭합니다.
4. 나타나는 UI 슬라이더에서 **Line plots**를 선택합니다.
5. **Point aggregation** 영역에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On hover** 또는 **Always**를 선택해주세요.
{{% /tab %}}

{{% tab header="워크스페이스 내 개별 차트" value="single_chart"%}}
1. 원하는 W&B 프로젝트로 이동하세요.
2. 왼쪽 탭에서 **Workspace** 아이콘을 클릭합니다.
3. full fidelity 모드를 적용하고 싶은 라인 플롯 패널을 선택합니다.
4. 나타나는 모달에서 **Show min/max values as a shaded area** 드롭다운 메뉴에서 **On hover** 또는 **Always**를 선택합니다.
{{% /tab %}}
{{< /tabpane >}}

### 데이터 정확성을 잃지 않고 데이터 탐색하기

데이터셋의 특정 부분을 분석할 때, 극단값이나 피크처럼 중요한 포인트를 놓치지 않을 수 있습니다. 라인 플롯을 확대(zoom)하면, W&B가 x-축 버킷 크기를 조정해서 해당 버킷 내에서 최소, 최대, 평균값을 다시 계산합니다.

{{< img src="/images/app_ui/zoom_in.gif" alt="Plot zoom functionality" >}}

기본적으로 W&B는 x-축을 1,000개의 버킷으로 동적으로 나눕니다. 각 버킷마다 W&B는 아래 값을 계산합니다:

- **Minimum**: 해당 버킷 내 가장 작은 값
- **Maximum**: 해당 버킷 내 가장 큰 값
- **Average**: 버킷 내 모든 포인트의 평균값

이렇게 버킷별로 값을 표현함으로써 데이터 전체 특성을 잃지 않으면서, 모든 플롯에 극단값도 포함할 수 있습니다. 확대하여 데이터 포인트 수가 1,000개 이하가 되면, full fidelity 모드에서는 별도의 집계 없이 모든 데이터를 직접 렌더링합니다.

라인 플롯을 확대하려면 다음 단계를 따라주세요:

1. 원하는 W&B 프로젝트로 이동하세요.
2. 왼쪽 탭에서 **Workspace** 아이콘을 클릭합니다.
3. 워크스페이스에 라인 플롯 패널을 추가하거나, 이미 있는 라인 플롯 패널로 이동합니다.
4. 확대하고 싶은 영역을 클릭한 후 드래그해 선택합니다.

{{% alert title="라인 플롯 그룹핑 및 익스프레션" %}}
Line Plot Grouping을 사용할 때, W&B는 선택한 모드에 따라 다음과 같이 작동합니다:

- **Non-windowed sampling (grouping)**: 여러 run의 포인트를 x-축상에 맞춰 정렬합니다. 여러 포인트가 같은 x-값을 가지면 평균값, 아니라면 별도의 포인트로 표시됩니다.
- **Windowed sampling (grouping and expressions)**: x-축을 250개의 버킷 또는 가장 긴 선의 포인트 수(더 작은 수)에 맞춰 나누고, 각 버킷 내 평균값을 계산합니다.
- **Full fidelity (grouping and expressions)**: non-windowed sampling과 비슷하지만, 각 run별로 최대 500개의 포인트만 읽어와 성능과 디테일 사이에 균형을 맞춥니다.
{{% /alert %}}

## Random sampling

Random sampling 모드는 1,500개의 포인트를 무작위로 샘플링하여 라인 플롯을 그립니다. 데이터 포인트가 매우 많을 때, 성능 저하를 막기 위해 유용하게 사용할 수 있습니다.

{{% alert color="warning" %}}
Random sampling은 비결정적으로 샘플링하기 때문에, 데이터의 중요한 이상치나 피크가 제외될 수 있으며, 그만큼 데이터의 정확성도 감소할 수 있습니다.
{{% /alert %}}

### Random sampling 활성화하기

기본적으로 W&B는 full fidelity 모드를 사용합니다. random sampling 모드를 사용하려면 아래 단계를 따라주세요:

{{< tabpane text=true >}}
{{% tab header="워크스페이스 내 모든 차트" value="all_charts" %}}
1. 원하는 W&B 프로젝트로 이동하세요.
2. 왼쪽 탭에서 **Workspace** 아이콘을 클릭합니다.
3. 화면 우측 상단의 **Add panels** 버튼 왼쪽에 위치한 톱니바퀴(설정) 아이콘을 클릭합니다.
4. 나타나는 UI 슬라이더에서 **Line plots**를 선택합니다.
5. **Point aggregation** 영역에서 **Random sampling**을 선택합니다.
{{% /tab %}}

{{% tab header="워크스페이스 내 개별 차트" value="single_chart"%}}
1. 원하는 W&B 프로젝트로 이동하세요.
2. 왼쪽 탭에서 **Workspace** 아이콘을 클릭합니다.
3. random sampling 모드를 적용하고 싶은 라인 플롯 패널을 선택합니다.
4. 나타나는 모달에서 **Point aggregation method** 영역에서 **Random sampling**을 선택합니다.
{{% /tab %}}
{{< /tabpane >}}

### 샘플링되지 않은 데이터 엑세스

[W&B Run API]({{< relref path="/ref/python/public-api/runs.md" lang="ko" >}})를 사용하면, run 도중 기록된 전체 메트릭 히스토리를 엑세스할 수 있습니다. 아래 예시는 특정 run에서 loss 값을 가져오고 처리하는 방법을 보여줍니다:

```python
# W&B API 초기화
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# 'Loss' 메트릭의 히스토리 불러오기
history = run.scan_history(keys=["Loss"])

# 히스토리에서 loss 값 추출
losses = [row["Loss"] for row in history]
```