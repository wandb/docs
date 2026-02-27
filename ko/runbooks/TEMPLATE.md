---
title: "템플릿"
---

<div id="agent-prompt-task-title">
  # 에이전트용 프롬프트: [Task title]
</div>

<div id="requirements">
  ## 요구 사항
</div>

이 작업을 시작하기 전에 충족해야 하는 액세스 요구 사항 또는 선행 조건을 나열하십시오:

- [ ] 필수 시스템 액세스(예: W&B 직원 액세스).
- [ ] 필수 권한(예: 저장소 쓰기 권한).
- [ ] 필수 도구 또는 종속성.

<div id="agent-prerequisites">
  ## 에이전트 사전 준비 사항
</div>

에이전트를 시작하기 전에 사용자로부터 미리 수집해야 하는 정보:

1. **[Required info 1]** - 필요한 이유
2. **[Required info 2]** - 필요한 이유
3. **[Optional info]** - 언제/왜 필요할 수 있는지

<div id="task-overview">
  ## 작업 개요
</div>

이 런북으로 수행할 작업과 사용 시점에 대한 간단한 설명을 작성합니다.

> **참고**: 사용자가 미리 알고 있어야 하는 중요한 컨텍스트나 제한 사항을 작성합니다.

<div id="context-and-constraints">
  ## 컨텍스트 및 제약 사항
</div>

<div id="systemtool-limitations">
  ### 시스템/도구 제한 사항
</div>

- 제한 사항 1과 이 제한 사항이 작업에 미치는 영향
- 제한 사항 2와 (있는 경우) 가능한 우회 방법

<div id="important-context">
  ### 중요한 맥락
</div>

- 핵심 배경 정보
- 자주 발생하는 함정이나 엣지 케이스
- 보안 관련 고려사항

<div id="step-by-step-process">
  ## 단계별 진행 방법
</div>

<div id="1-first-major-step">
  ### 1. [첫 번째 주요 단계]
</div>

이 단계에서 어떤 작업이 수행되는지에 대한 설명입니다.

```bash
# 예시 명령어
command --with-flags
```

**예상 결과**: 이 단계를 완료한 후에 기대되는 동작입니다.


<div id="2-second-major-step">
  ### 2. [두 번째 주요 단계]
</div>

설명과 의사 결정 포인트.

**에이전트 메모**: AI 에이전트를 위한 특별 지침 예시:

- 사용자에게 추가 설명을 요청해야 하는 시점
- 권한이 부족할 때 사용할 폴백 절차
- 일반적인 변형 상황을 처리하는 방법

<div id="3-continue-with-remaining-steps">
  ### 3. [나머지 단계를 계속 진행합니다...]
</div>

<div id="verification-and-testing">
  ## 검증 및 테스트
</div>

예상 결과:

- ✓ 성공 기준 1
- ✓ 성공 기준 2
- ✗ 일반적인 실패 징후와 그 의미

<div id="how-to-verify-success">
  ### 성공 여부 확인 방법
</div>

1. 다음을 확인하세요...
2. 다음이 충족되는지 확인하세요...
3. 다음과 같이 테스트하세요...

<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-common-problem-1">
  ### 문제: [Common problem 1]
</div>

- **Symptoms**: 문제가 어떻게 나타나는지
- **Cause**: 발생 원인
- **Solution**: 단계별 해결 절차

<div id="issue-common-problem-2">
  ### 문제: [Common problem 2]
</div>

- **증상**: 
- **원인**: 
- **해결 방법**: 

<div id="cleanup-instructions">
  ## 정리 지침
</div>

작업을 완료한 후에는:

1. 임시 파일과 브랜치를 삭제합니다.
2. 수정된 설정을 모두 초기화합니다.
3. 수행한 영구적 변경 사항을 기록으로 남깁니다.

```bash
# 정리 명령어 예시
git branch -D temp-branch-name
rm -f temporary-files
```


<div id="checklist">
  ## 체크리스트
</div>

전체 프로세스를 위한 요약 체크리스트:

- [ ] 모든 요구 사항을 충족했는지 확인했다.
- [ ] 사용자로부터 필요한 정보를 수집했다.
- [ ] 1단계를 완료했다: [간단한 설명].
- [ ] 2단계를 완료했다: [간단한 설명].
- [ ] 결과를 검증했다.
- [ ] 임시 리소스를 정리했다.
- [ ] 영구적인 변경 사항을 문서화했다.

<div id="notes">
  ## 참고 사항
</div>

- 추가 팁이나 맥락.
- 관련 문서 링크.
- 다른 접근 방식을 사용해야 하는 경우.