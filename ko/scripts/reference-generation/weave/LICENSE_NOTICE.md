---
title: 라이선스 고지
---

<div id="license-notice-for-reference-documentation-generation">
  # 레퍼런스 문서 생성을 위한 라이선스 안내
</div>

<div id="overview">
  ## 개요
</div>

이 디렉터리에 있는 스크립트는 개발/CI 과정에서 레퍼런스 문서를 생성하는 용도로만 사용됩니다. 이 스크립트는 Weave 라이브러리와 함께 배포되지 않으며, 어떠한 프로덕션 코드에도 포함되지 않습니다.

<div id="dependencies-and-their-licenses">
  ## 의존성과 해당 라이선스
</div>

<div id="direct-dependencies">
  ### 직접 의존성
</div>

- **requests** (Apache-2.0): HTTP 요청을 보내는 데 사용
- **lazydocs** (MIT): W&B에서 관리하는 문서 생성 도구

<div id="transitive-dependencies-via-lazydocs">
  ### 전이적 의존성(lazydocs를 통해)
</div>

- **setuptools** (일부 포함된 컴포넌트는 LGPL-3.0인 MIT 라이선스): 빌드 시스템
- 기타 여러 의존성(혼합 라이선스)

<div id="important-notes">
  ## 중요 안내
</div>

1. **개발 용도 전용**: 이 의존성들은 CI/GitHub Actions에서 문서를 생성하는 동안에만 일시적으로 설치되며, 배포되는 Weave 패키지에는 절대 포함되지 않습니다.

2. **배포 대상 아님**: 생성된 문서는 실행 가능한 코드나 의존성 없이 MDX/Markdown 파일만으로 구성됩니다.

3. **격리된 실행**: GitHub Action은 이러한 스크립트를 사용 후 삭제되는 격리된 가상 환경에서 실행합니다.

4. **라이선스 준수**: 이러한 도구들은 Weave와 함께 배포되지 않으므로, setuptools의 내장 의존성에 포함된 LGPL-3.0 구성 요소는 Weave 사용자에게 추가적인 라이선스 의무를 부과하지 않습니다.

<div id="for-organizations-with-strict-license-policies">
  ## 라이선스 정책이 엄격한 조직의 경우
</div>

소속 조직에 개발 도구에서 LGPL 코드 사용을 금지하는 정책이 있는 경우:

1. GitHub Action을 사용해 클라우드에서 문서를 생성합니다(권장)
2. lazydocs를 사용하지 않는 최소한의 Python 문서 생성기를 사용합니다
3. Docker 컨테이너에서 문서를 생성합니다
4. 개발 전용 도구에 대해 예외 승인을 요청합니다

<div id="socket-security">
  ## 소켓 보안
</div>

저장소 루트에 있는 `.socketignore` 파일은 이 스크립트들이 프로덕션 코드가 아닌 개발 도구이므로 보안 검사 대상에서 제외합니다.

<div id="known-socket-security-warnings">
  ### 알려진 소켓 보안 경고
</div>

- **wheel 안의 네이티브 코드**: `wheel` 패키지에는 네이티브 코드가 포함되어 있으며, 이는 Python 패키징 도구에서는 정상적인 현상입니다.
- **라이선스 위반**: 일부 전이적 의존성에 LGPL 등 정책 경고를 유발하는 라이선스가 포함되어 있을 수 있습니다.

이러한 경고는 다음과 같은 이유로 허용됩니다.

1. 해당 도구들은 문서 생성 과정에서만 사용됩니다.
2. 격리된 CI 환경에서만 실행됩니다.
3. Weave와 함께 배포되지 않습니다.
4. 생성된 문서에는 실행 가능한 코드가 포함되지 않습니다.