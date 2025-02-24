---
title: Compare runs across projects
description: 크로스 프로젝트 리포트로 두 개의 다른 프로젝트의 run을 비교해보세요.
menu:
  default:
    identifier: ko-guides-core-reports-cross-project-reports
    parent: reports
weight: 60
---

크로스-project reports를 사용하여 서로 다른 두 project의 runs를 비교해보세요. run set 테이블에서 project 선택기를 사용하여 project를 선택하세요.

{{< img src="/images/reports/howto_pick_a_different_project_to_draw_runs_from.gif" alt="Compare runs across different projects" >}}

섹션의 시각화는 첫 번째 활성 runset에서 열을 가져옵니다. 라인 플롯에서 찾고 있는 메트릭이 보이지 않으면 섹션에서 처음으로 확인된 run set에 해당 열이 있는지 확인하세요.

이 기능은 시계열 라인에서 history 데이터를 지원하지만 서로 다른 project에서 다른 요약 메트릭을 가져오는 것은 지원하지 않습니다. 즉, 다른 project에서만 로그된 열에서 산점도를 만들 수 없습니다.

두 project의 runs를 비교해야 하는데 열이 작동하지 않으면 한 project의 runs에 태그를 추가한 다음 해당 runs를 다른 project로 이동하세요. 각 project의 runs만 필터링할 수 있지만 report에는 두 sets의 runs에 대한 모든 열이 포함됩니다.

## 보기 전용 report 링크

private project 또는 team project에 있는 report에 대한 보기 전용 링크를 공유하세요.

{{< img src="/images/reports/magic-links.gif" alt="" >}}

보기 전용 report 링크는 URL에 비밀 access 토큰을 추가하므로 링크를 여는 모든 사람이 페이지를 볼 수 있습니다. 누구나 magic 링크를 사용하여 먼저 로그인하지 않고도 report를 볼 수 있습니다. [W&B Local]({{< relref path="/guides/hosting/" lang="ko" >}}) 프라이빗 cloud 설치를 사용하는 고객의 경우 이러한 링크는 방화벽 뒤에 유지되므로 프라이빗 인스턴스에 대한 access 권한과 보기 전용 링크에 대한 access 권한이 있는 team 구성원만 report를 볼 수 있습니다.

**보기 전용 모드**에서는 로그인하지 않은 사람이 차트를 보고 마우스를 올려 값의 툴팁을 보고, 차트를 확대 및 축소하고, 테이블에서 열을 스크롤할 수 있습니다. 보기 모드에서는 데이터를 탐색하기 위해 새 차트나 새 테이블 쿼리를 만들 수 없습니다. report 링크의 보기 전용 방문자는 run을 클릭하여 run 페이지로 이동할 수 없습니다. 또한 보기 전용 방문자는 공유 모달을 볼 수 없지만 대신 마우스를 올리면 `보기 전용 access에는 공유를 사용할 수 없습니다`라는 툴팁이 표시됩니다.

{{% alert color="info" %}}
magic 링크는 "Private" 및 "Team" projects에서만 사용할 수 있습니다. "Public"(누구나 볼 수 있음) 또는 "Open"(누구나 runs를 보고 기여할 수 있음) projects의 경우 이 project는 이미 링크가 있는 모든 사람이 사용할 수 있음을 의미하므로 링크를 켜거나 끌 수 없습니다.
{{% /alert %}}

## report에 그래프 보내기

진행 상황을 추적하기 위해 워크스페이스에서 report로 그래프를 보냅니다. report에 복사하려는 차트 또는 패널의 드롭다운 메뉴를 클릭하고 **report에 추가**를 클릭하여 대상 report를 선택합니다.
