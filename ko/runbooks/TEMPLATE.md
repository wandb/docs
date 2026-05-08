---
title: TEMPLATE
---

<div id="agent-prompt-task-title">
  # Agent 프롬프트: [작업 제목]
</div>

<div id="requirements">
  ## 요구 사항
</div>

이 작업을 시작하기 전에 충족해야 하는 액세스 요구 사항이나 사전 요구 사항을 목록으로 작성합니다:

* [ ] 필수 시스템 액세스 권한(예: W&amp;B 직원 액세스 권한)
* [ ] 필수 권한(예: 저장소 쓰기 액세스 권한)
* [ ] 필수 도구 또는 의존성

<div id="agent-prerequisites">
  ## 에이전트 사전 요구 사항
</div>

시작하기 전에 사용자로부터 확인해야 할 정보:

1. **[필수 정보 1]** - 왜 필요한지
2. **[필수 정보 2]** - 왜 필요한지
3. **[선택 정보]** - 언제/왜 필요할 수 있는지

<div id="task-overview">
  ## 작업 Overview
</div>

이 runbook이 무엇을 수행하는지와 언제 사용해야 하는지를 간략히 설명합니다.

> **참고**: 사용자가 미리 알아두어야 할 중요한 맥락이나 제한 사항입니다.

<div id="context-and-constraints">
  ## 맥락 및 제약 사항
</div>

<div id="systemtool-limitations">
  ### 시스템/도구 제한 사항
</div>

* 제한 사항 1과 이것이 작업에 미치는 영향
* 제한 사항 2와 가능한 경우의 해결 방법

<div id="important-context">
  ### 중요 배경
</div>

* 핵심 배경 정보
* 흔히 놓치기 쉬운 부분이나 예외 사례
* 보안 고려 사항

<div id="step-by-step-process">
  ## Step-by-step 프로세스
</div>

<div id="1-first-major-step">
  ### 1. [첫 번째 주요 step]
</div>

이 step에서 수행하는 작업을 설명합니다.

```bash
# 예시 명령어
command --with-flags
```

**예상 결과**: 이 step 후에 어떤 일이 일어나야 하는지 설명합니다.


<div id="2-second-major-step">
  ### 2. [두 번째 주요 step]
</div>

설명과 필요한 의사결정 지점.

**에이전트 참고**: 다음과 같은 AI agent용 특별 지침:

* 언제 사용자에게 추가 설명을 요청해야 하는지
* 권한이 부족할 때의 대체 절차
* 일반적인 변형을 처리하는 방법

<div id="3-continue-with-remaining-steps">
  ### 3. [남은 step을 계속 진행...]
</div>

<div id="verification-and-testing">
  ## 검증 및 테스트
</div>

예상되는 결과:

* ✓ 성공 확인 항목 1
* ✓ 성공 확인 항목 2
* ✗ 일반적인 실패 징후와 그 의미

<div id="how-to-verify-success">
  ### 성공했는지 확인하는 방법
</div>

1. 다음을 확인합니다...
2. 다음이 맞는지 확인합니다...
3. 다음과 같이 테스트합니다...

<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-common-problem-1">
  ### 문제: [일반적인 문제 1]
</div>

* **증상**: 이 문제가 어떻게 나타나는지
* **원인**: 왜 발생하는지
* **해결 방법**: step별 해결 방법

<div id="issue-common-problem-2">
  ### 문제: [자주 발생하는 문제 2]
</div>

* **증상**:
* **원인**:
* **해결 방법**: 

<div id="cleanup-instructions">
  ## 정리 지침
</div>

작업을 완료한 후:

1. 임시 파일/브랜치를 제거합니다.
2. 변경한 설정을 재설정합니다.
3. 영구적으로 변경한 사항을 문서화합니다.

```bash
# 정리 명령어 예시
git branch -D temp-branch-name
rm -f temporary-files
```


<div id="checklist">
  ## 체크리스트
</div>

전체 프로세스용 요약 체크리스트:

* [ ] 모든 요구사항을 충족했습니다.
* [ ] 사용자에게서 필요한 정보를 수집했습니다.
* [ ] step 1을 완료했습니다: [간단한 설명].
* [ ] step 2를 완료했습니다: [간단한 설명].
* [ ] 결과를 확인했습니다.
* [ ] 임시 리소스를 정리했습니다.
* [ ] 영구적인 변경 사항을 문서화했습니다.

<div id="notes">
  ## 참고 사항
</div>

* 추가 팁이나 맥락
* 관련 문서 링크
* 대체 접근 방식을 사용해야 하는 경우