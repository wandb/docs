---
title: Locadex AI 컨텍스트 설정
---

<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # Agent 프롬프트: W&amp;B 문서용 Locadex AI 컨텍스트 설정(한국어, 이후 일본어)
</div>

<div id="requirements">
  ## 요구 사항
</div>

* [ ] [General Translation Dashboard](https://dash.generaltranslation.com/) (Locadex 콘솔) 액세스 권한.
* [ ] Locadex/GT 프로젝트에 연결된 docs 저장소(GitHub 앱이 설치되어 있고 저장소가 연결되어 있어야 함).
* [ ] 선택 사항: 용어집이나 로캘 컨텍스트를 다듬을 때 수동 번역본을 비교할 수 있도록, `ko/`(선택적으로 `ja/`도 포함)가 있는 wandb/docs의 `main` 브랜치 액세스 권한.

<div id="agent-prerequisites">
  ## Agent 사전 요구 사항
</div>

1. **어떤 로캘을 설정하려고 하나요?** (예: 지금은 한국어만, 일본어는 나중에.) 이를 기준으로 추가할 Glossary 번역과 Locale Context 항목이 결정됩니다.
2. **이미 Glossary CSV 또는 용어 목록이 있나요?** 없다면 runbook을 사용해 아래 소스에서 직접 만드세요.
3. **GT 프로젝트가 이미 생성되어 있고 repo가 연결되어 있나요?** 아니라면 먼저 [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify)의 1~6단계를 완료하세요.

<div id="task-overview">
  ## 작업 개요
</div>

이 runbook에서는 (1) 기존 `wandb_docs_translation` 도구와 (2) `main`에 있는 수동 번역 한국어 콘텐츠(이후 일본어 포함)에서 번역 메모리와 용어를 추출하는 방법, 그리고 자동 번역이 이 컨텍스트를 활용하도록 Locadex/General Translation 플랫폼을 설정하는 방법을 설명합니다. 목표는 제품명과 기술 용어에 대해 일관된 용어를 사용하고, “번역하지 않음” 규칙이 올바르게 적용되도록 하는 것입니다.

**구성 위치:**

| What                                   | Where                                                        | Notes                                                                 |
| -------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------- |
| **Glossary** (용어, 정의, 로캘별 번역)          | Locadex console → AI Context → Glossary                      | 일관된 용어 사용과 제품/기능 이름에 대한 “번역하지 않음” 규칙을 적용합니다. CSV로 일괄 업로드할 수 있습니다.     |
| **Locale Context** (언어별 지침)            | Locadex console → AI Context → Locale Context                | 예: 한국어의 경우 알파벳과 한글 사이 띄어쓰기, 서식 규칙                                     |
| **Style Controls** (톤, 대상 독자, 프로젝트 설명) | Locadex console → AI Context → Style Controls                | 프로젝트 전체에 적용되며 모든 로캘에 공통으로 적용됩니다.                                      |
| **번역할 파일/로캘**                          | Git → `gt.config.json`                                       | `locales`, `defaultLocale`, `files`. 리포지토리에는 glossary나 프롬프트를 두지 않습니다. |
| **벤더 이슈 로그 (Locadex 버그)**              | Git → [locadex-vendor-issues.md](./locadex-vendor-issues.md) | Locadex에 보고할 번역 결함을 추적합니다(예: 현지화된 MDX에서 URL이 깨지는 문제).                 |

즉, **자동 번역은 Locadex console에서 제어합니다** (Glossary, Locale Context, Style Controls). **파일 및 로캘 설정은 Git에 유지합니다** (`gt.config.json`). `gt.config.json`의 선택적 `dictionary` 키는 문서 MDX glossary용이 아니라 앱 UI 문자열(예: gt-next/gt-react)용이며, 문서 용어는 console에서 관리합니다.

<div id="context-and-constraints">
  ## 맥락 및 제약 사항
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### 레거시 도구 (wandb_docs_translation)
</div>

* **human&#95;prompt.txt**: 절대로 번역하면 안 되는 W&amp;B 제품/기능 이름을 나열합니다(영문 유지): Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models. `[**word**](link)`와 같은 링크/목록 문맥에도 동일하게 적용됩니다.
* **system&#95;prompt.txt**: 일반 규칙(유효한 Markdown, 코드 블록에서는 주석만 번역, 사전 사용, 링크 URL은 번역하지 않음, 일본어/한국어의 경우 알파벳과 CJK 문자 사이, 그리고 인라인 서식 주변에 공백 추가).
* **configs/language&#95;dicts/ko.yaml**: 혼합된 “번역 메모리”:
  * **영문 유지** (제품/기능 이름): 예: `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`.
  * **한국어로 번역**: 예: `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자.

따라서 관례는 다음과 같았습니다: **제품/기능 이름(대개 첫 글자가 대문자이거나 UI/목록 문맥에 있는 경우)은 영어로 유지**하고, **보통명사 용법**은 로캘 사전을 따릅니다. Locadex Glossary는 각 로캘별로 “번역하지 않음”과 “X로 번역”을 모두 반영해야 합니다.

<div id="locadexgt-platform-behavior">
  ### Locadex/GT 플랫폼 동작 방식
</div>

* **Glossary**: 용어(원문 기준) + 선택적 정의 + 선택적 로캘별 번역으로 구성됩니다. “번역하지 않음”인 경우 해당 로캘에서 용어와 동일한 문자열을 사용합니다(예: 용어 “W&amp;B”, 번역 (ko) “W&amp;B”). “다음과 같이 번역”인 경우 Translation (ko)에 원하는 대상어를 설정합니다(예: “artifact” → “아티팩트”).
* **Locale Context**: 대상 로캘별 자유 형식의 지침입니다(예: “라틴 문자와 한글 사이에 공백 사용”).
* **Style Controls**: 프로젝트별로 하나의 세트(톤, 대상 독자, 설명)입니다. 모든 로캘에 적용됩니다.
* AI Context 변경 사항은 기존 콘텐츠를 자동으로 재번역하지 **않습니다**. 이미 번역된 파일에 새 컨텍스트를 적용하려면 [Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate)를 사용하세요.

<div id="step-by-step-process">
  ## step별 절차
</div>

<div id="1-gather-terminology-sources">
  ### 1. 용어 출처 수집
</div>

* **wandb&#95;docs&#95;translation에서** (가능한 경우):
  * `configs/human_prompt.txt` → 절대 번역하면 안 되는 용어 목록
  * `configs/language_dicts/ko.yaml` (이후에는 `ja.yaml`도) → 용어 → 로캘별 번역 매핑
* **main의 수동 번역본에서** (선택): EN과 KO(또는 JA) 페이지 몇 개를 비교해 제품명과 자주 쓰이는 용어가 어떻게 번역되었는지 확인하고(예: “run” 대 “실행”, “workspace” 대 “워크스페이스”), 용어집 항목을 추가하거나 조정합니다.

**Agent 참고**: Agent가 외부 리포지토리를 읽을 수 없더라도, 이 리포지토리에서 제공하는 CSV와 로캘 컨텍스트 텍스트를 사용해 사람이 계속 이 runbook을 따를 수 있습니다(아래의 runbooks와 선택 사항인 CSV 참조).

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. Glossary CSV를 빌드하거나 획득하기
</div>

* 이 리포지토리의 한국어용 사전 빌드된 Glossary CSV를 사용하세요: **runbooks/locadex-glossary-ko.csv**(아래 “Glossary CSV” 참조). 또는 다음을 포함하는 파일을 생성하세요:
  * **번역하지 않을 용어**: 용어당 한 행, Definition은 선택 사항, `ko`(또는 “Translation (ko)”)는 Term과 동일하게 설정합니다.
  * **번역할 용어**: 용어당 한 행, Definition은 선택 사항, `ko`에는 원하는 한국어 대응어를 입력합니다.
* Locadex “Upload Context CSV”에서 요구하는 정확한 column 이름(예: `Term`, `Definition`, `ko` 또는 `Translation (ko)`)을 확인하세요. 콘솔이 다른 이름을 요구하면 CSV 헤더를 조정하세요.
* **CSV 형식(올바르게 파싱되도록)**: 파일이 올바르게 파싱되도록 표준 CSV 인용 규칙을 사용하세요. 쉼표는 필드 구분자입니다. 쉼표, 큰따옴표 또는 줄바꿈이 포함된 필드는 **반드시** 큰따옴표로 감싸야 합니다. 큰따옴표로 감싼 필드 안에서 큰따옴표를 사용해야 하면 큰따옴표를 두 번 써서 이스케이프하세요(`""`). 용어는 한 행에 하나씩만 넣으세요(“run, Run”처럼 여러 변형을 한 셀에 넣지 마세요). 프로그래밍 방식으로 CSV를 생성하거나 편집할 때는 CSV 라이브러리를 사용하거나 해당 필드를 명시적으로 인용하세요. Term 또는 Definition에 따옴표로 감싸지 않은 쉼표가 있으면 column 경계로 처리되어 행이 깨집니다.

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. 콘솔에서 Locadex 프로젝트 설정하기
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/)에 로그인합니다.
2. wandb/docs 저장소에 연결된 프로젝트를 엽니다.
3. **AI Context**로 이동합니다(또는 해당 항목인 Glossary, Locale Context, Style Controls로 이동).

<div id="4-upload-or-add-glossary-terms">
  ### 4. Glossary 용어 업로드 또는 추가
</div>

* **옵션 A**: **Upload Context CSV**를 사용해 Glossary를 일괄 임포트합니다(Term, Definition, 로캘 열). 플랫폼이 열을 Glossary 용어와 로캘별 번역에 매핑합니다.
* **옵션 B**: 용어를 수동으로 추가합니다: Term, Definition(모델에 도움을 줌), 그리고 한국어의 경우 번역을 추가합니다(“번역하지 않음”은 term과 동일하게, “다음과 같이 번역”은 한국어 문자열로 입력).

최소한 다음은 포함하세요:

* 영어로 유지해야 하는 제품/기능 이름: W&amp;B, Weights &amp; Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models 등. 한국어 값은 소스와 동일하게 설정합니다.
* 일관되게 번역해야 하는 용어: 예: artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스, 그리고 `language_dicts/ko.yaml`의 다른 항목들(이후 `ja.yaml`도 동일).

<div id="5-set-locale-context-for-korean">
  ### 5. 한국어용 Locale Context 설정
</div>

* 로캘 **ko**를 Select합니다.
* 기존 system&#95;prompt와 한국어 문서의 모범 사례를 반영하는 지침을 추가합니다. 예를 들면 다음과 같습니다:
  * 라틴 문자와 한국어 문자(한글, 한자 포함)를 함께 쓸 때는 사이에 공백을 넣습니다.
  * 한국어 단어나 구문의 일부에 인라인 서식(굵게, 기울임꼴, 코드)을 사용할 때는 markdown가 올바르게 render되도록 서식이 적용된 부분의 앞뒤에 공백을 넣습니다.
  * 코드 블록과 링크 URL은 그대로 두고, 주변 설명문과 필요한 경우 코드 내 주석만 번역합니다.

Locale Context를 Save합니다.

<div id="6-set-style-controls-project-wide">
  ### 6. Style Controls 설정(프로젝트 전체에 적용)
</div>

* **프로젝트 설명**: 예: “Weights &amp; Biases (W&amp;B) 문서: ML 실험 추적, 모델 레지스트리, LLM Ops용 Weave 및 관련 제품.”
* **대상 독자**: 개발자와 ML 실무자.
* **톤**: 전문적이고 기술적이며 명확한 어조. 직역보다 자연스러운 표현을 우선합니다.

저장.

<div id="7-retranslate-if-needed">
  ### 7. 필요한 경우 다시 번역하기
</div>

* 이미 자동 번역된 콘텐츠가 있고 Glossary 또는 Locale Context를 변경했다면, 영향을 받는 파일에 대해 플랫폼의 **Retranslate** 흐름을 사용해 새 컨텍스트가 적용되도록 하세요.

<div id="verification-and-testing">
  ## 검증 및 테스트
</div>

* **Glossary**: 업로드 후 Glossary 탭에서 몇 가지 용어를 골라 확인합니다(번역 금지 항목과 번역된 항목).
* **Locale Context**: 한국어(그리고 이후 일본어) 지침이 올바른 로캘에 저장되었는지 확인합니다.
* **Quality**: 샘플 페이지에서 번역을 실행하거나 트리거한 뒤, 제품명이 영어로 유지되는지, 일반 용어가 용어집과 일치하는지 확인합니다(예: 적절한 경우 artifact → 아티팩트).

<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 문제: CSV upload가 Glossary에 매핑되지 않음
</div>

* **원인**: 열 이름이 플랫폼에서 기대하는 값과 일치하지 않을 수 있습니다.
* **해결 방법**: “Upload Context CSV” 열 이름(예: Term, Definition, locale code)은 Locadex/GT 문서나 UI 도움말에서 확인하세요. CSV의 열 이름을 수정한 뒤 다시 upload하세요.

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### 문제: 영어로 유지해야 하는 용어가 계속 번역됨
</div>

* **원인**: 용어가 Glossary에 없거나 “번역하지 않음” 설정이 되어 있지 않음(해당 로캘 번역이 누락되었거나 잘못됨).
* **해결 방법**: 대상 로캘에도 같은 값으로 해당 용어를 Glossary에 추가합니다(예: “Artifacts” → ko: “Artifacts”). 모델이 이것이 제품/기능 이름이라는 점을 이해할 수 있도록 짧은 Definition도 추가합니다.

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 문제: 일본어(또는 다른 로캘)에는 별도의 규칙이 필요합니다
</div>

* **원인**: 로캘별 선호 사항(예: 존댓말, 공백 처리, 제품 이름의 가타카나 표기).
* **해결 방법**: 해당 로캘(예: ja)에 대한 별도의 Locale Context를 추가하고, 필요한 경우 “ja” 열이 포함된 Glossary 항목을 추가하거나 일본어용 수동 항목을 추가합니다.

<div id="cleanup-instructions">
  ## 정리 지침
</div>

* 콘솔 전용 설정의 경우 docs 리포지토리에 임시 브랜치나 파일이 필요하지 않습니다.
* CSV를 생성하기 위한 일회성 스크립트를 만들었다면, 팀에서 유지하기로 결정한 경우가 아니라면 커밋하지 마세요(일회성 스크립트에 관한 AGENTS.md 및 사용자 규칙 참조).

<div id="checklist">
  ## 체크리스트
</div>

* [ ] human&#95;prompt, language&#95;dicts/ko.yaml(해당하는 경우 ja도 포함)에서 용어를 수집했다.
* [ ] 업로드용 Glossary CSV를 작성하거나 구해 column 이름을 확인했다.
* [ ] Locadex 콘솔에 로그인하고 올바른 프로젝트를 열었다.
* [ ] Glossary 용어(번역 금지 항목 및 번역된 항목)를 업로드하거나 추가했다.
* [ ] 한국어용 Locale Context를 설정했다(해당하는 경우 이후 일본어도 포함).
* [ ] Style Controls(설명, audience, tone)를 설정했다.
* [ ] 샘플 번역으로 확인하고 필요하면 기존 콘텐츠를 다시 번역했다.

<div id="glossary-csv">
  ## Glossary CSV
</div>

이 리포지토리에는 한국어 용어집 시작본이 제공됩니다: **runbooks/locadex-glossary-ko.csv**. 열은 다음과 같습니다.

* **Term**: 문서에 나오는 원문(영어) 용어입니다.
* **Definition**: 짧은 설명입니다(AI에 도움이 되며, 업로드 시에는 선택 사항).
* **ko**: 한국어 번역입니다. “번역하지 않음”인 경우 Term과 동일한 문자열을 사용하고, “다음과 같이 번역”인 경우 원하는 한국어 문자열을 사용합니다.

`configs/language_dicts/ko.yaml`(또는 main의 수동 KO 페이지)에서 용어를 더 추가하려면, 동일한 열로 행을 덧붙이세요. Locadex 콘솔이 로캘 번역에 대해 다른 열 이름(예: “Translation (ko)”)을 요구하는 경우, 업로드할 때 또는 업로드 전에 CSV에서 `ko` 열 이름을 변경하세요.

<div id="csv-formatting-for-future-generation">
  ### 향후 생성을 위한 CSV 형식
</div>

용어집 CSV를 생성하거나 여기에 내용을 추가할 때(수동 또는 스크립트 사용) 파일이 올바른 형식을 유지하도록 다음 규칙을 따르세요:

* **구분자**: 쉼표(`,`)를 사용합니다. 필드가 따옴표로 묶여 있지 않으면 필드 안에 쉼표를 사용하지 마세요.
* **따옴표 처리**: 필드에 쉼표, 큰따옴표 또는 줄바꿈이 포함된 경우 해당 필드를 큰따옴표(`"`)로 감싸세요. 일관성을 위해 모든 필드를 큰따옴표로 감싸도 됩니다.
* **이스케이프**: 따옴표로 감싼 필드 안에서 큰따옴표 자체는 큰따옴표 두 개(`""`)로 표시합니다.
* **행당 하나의 용어**: 각 행에는 하나의 용어만 넣습니다. 하나의 셀에 여러 변형을 나열하지 마세요(예: Term 열에 “run, artifact”라고 쓰지 말고 “run”과 “artifact”를 각각 별도 행에 넣으세요).
* **도구**: 프로그래밍 방식으로 CSV를 생성할 때는 올바른 CSV 라이브러리(예: Python `csv` 모듈에서 `quoting=csv.QUOTE_MINIMAL` 또는 `QUOTE_NONNUMERIC` 사용)를 사용하여 Term 또는 Definition에 있는 쉼표와 따옴표가 올바르게 처리되도록 하세요.

<div id="notes">
  ## 참고 사항
</div>

* **일본어는 나중에 추가**: 일본어를 추가할 때는 `ja`의 Locale Context를 다시 명시하고(예: 공손한 표현, 알파벳과 일본어 문자 사이 공백, 인라인 서식 공백), `ja`용 Glossary 항목도 추가하세요(같은 방식: 번역하지 않음 = 원문과 동일, 다음과 같이 번역 = 원하는 일본어).
* **Git의 GT 설정**: `gt.config.json`에는 이미 `locales`와 `defaultLocale`가 있습니다. glossary나 AI context는 여기에 저장되지 않고 콘솔에만 있습니다.
* **레퍼런스**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary), [Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).