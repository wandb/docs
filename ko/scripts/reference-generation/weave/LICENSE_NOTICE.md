---
title: 라이선스 고지
---

<div id="license-notice-for-reference-documentation-generation">
  # 레퍼런스 문서 생성을 위한 라이선스 고지
</div>

<div id="overview">
  ## Overview
</div>

이 디렉터리의 스크립트는 개발/CI 과정에서 레퍼런스 문서를 생성하는 용도로만 사용됩니다. Weave 라이브러리와 함께 배포되지 않으며, 프로덕션 코드에도 포함되지 않습니다.

<div id="dependencies-and-their-licenses">
  ## 의존성과 해당 라이선스
</div>

<div id="direct-dependencies">
  ### 직접 의존성
</div>

* **requests** (Apache-2.0): HTTP 요청에 사용
* **lazydocs** (MIT): W&amp;B에서 유지 관리하는 문서 생성기

<div id="transitive-dependencies-via-lazydocs">
  ### 전이적 의존성 (via lazydocs)
</div>

* **setuptools** (벤더링된 LGPL-3.0 컴포넌트가 포함된 MIT): 빌드 시스템
* 서로 다른 라이선스가 적용되는 기타 여러 의존성

<div id="important-notes">
  ## 중요 참고 사항
</div>

1. **개발 전용**: 이러한 의존성은 CI/GitHub Actions에서 문서를 생성하는 동안에만 일시적으로 설치됩니다. 배포되는 Weave 패키지에는 절대 포함되지 않습니다.

2. **배포되지 않음**: 생성된 문서는 실행 가능한 코드나 의존성 없이 MDX/Markdown 파일로만 이루어집니다.

3. **격리된 실행**: GitHub Action은 이러한 스크립트를 격리된 가상 환경에서 실행하며, 해당 환경은 사용 후 삭제됩니다.

4. **라이선스 준수**: 이러한 도구는 Weave와 함께 배포되지 않으므로, setuptools에 벤더링된 의존성의 LGPL-3.0 컴포넌트로 인해 Weave 사용자에게 라이선스 의무가 발생하지 않습니다.

<div id="for-organizations-with-strict-license-policies">
  ## 엄격한 라이선스 정책이 있는 조직의 경우
</div>

조직에서 개발 도구에 LGPL 코드가 포함되는 것을 허용하지 않는 정책이 있는 경우:

1. GitHub Action을 사용해 클라우드에서 문서를 생성합니다(권장)
2. lazydocs를 사용하지 않는 최소 Python generator를 사용합니다
3. 도커 컨테이너에서 문서를 생성합니다
4. 개발 전용 도구에 대한 예외를 요청합니다

<div id="socket-security">
  ## Socket Security
</div>

저장소 루트의 `.socketignore` 파일은 이 스크립트들이 프로덕션 코드가 아니라 개발 도구이기 때문에 보안 스캔에서 제외합니다.

<div id="known-socket-security-warnings">
  ### 알려진 Socket Security 경고
</div>

* **wheel의 네이티브 코드**: `wheel` 패키지에는 네이티브 코드가 포함되어 있는데, 이는 Python 패키징 도구에서는 정상입니다
* **라이선스 위반**: 일부 전이적 의존성에는 정책 경고를 트리거하는 LGPL 또는 기타 라이선스가 포함될 수 있습니다

이러한 경고를 허용할 수 있는 이유는 다음과 같습니다:

1. 이러한 도구는 문서 생성 시에만 사용됩니다
2. 격리된 CI 환경에서 실행됩니다
3. Weave와 함께 배포되지 않습니다
4. 생성된 문서에는 실행 가능한 코드가 없습니다