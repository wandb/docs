---
title: Locadex Ai 컨텍스트 설정
---

<div id="agent-prompt-configure-locadex-ai-context-for-wb-docs-korean-and-later-japanese">
  # 에이전트용 프롬프트: W&B 문서를 위한 Locadex AI 컨텍스트 설정 (한국어, 추후 일본어)
</div>

<div id="requirements">
  ## 요구 사항
</div>

- [ ] [General Translation Dashboard](https://dash.generaltranslation.com/) (Locadex 콘솔)에 대한 접근 권한.
- [ ] Locadex/GT 프로젝트에 연결된 문서 리포지토리(GitHub 앱 설치 및 리포지토리 연결 완료).
- [ ] 선택 사항: [hw-wandb/wandb_docs_translation](https://github.com/hw-wandb/wandb_docs_translation) 리포지토리에 대한 읽기 권한(구성 및 language_dicts 확인용).
- [ ] 선택 사항: wandb/docs의 `main` 브랜치에 존재하는 `ko/`(및 선택적으로 `ja/`)에 대한 접근 권한으로, 용어집 또는 로케일 컨텍스트를 다듬을 때 수동 번역과 비교하는 데 사용.

<div id="agent-prerequisites">
  ## 에이전트 사전 준비 사항
</div>

1. **어떤 로케일을 설정할 예정인가요?** (예: 지금은 Korean만, 이후 Japanese.) 이에 따라 어떤 Glossary 번역과 Locale Context 항목을 추가할지 결정합니다.
2. **이미 Glossary CSV 또는 용어 목록이 있나요?** 아직 없다면 아래 소스를 참고해 런북을 사용해 새로 만드세요.
3. **GT 프로젝트가 이미 생성되어 있고 레포가 연결되어 있나요?** 아니라면 먼저 [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify)의 1–6단계를 완료하세요.

<div id="task-overview">
  ## 작업 개요
</div>

이 런북에서는 (1) 기존 `wandb_docs_translation` 툴링과 (2) `main` 브랜치에 수동으로 번역된 한국어(이후에는 일본어) 콘텐츠에서 번역 메모리와 용어를 추출하는 방법, 그리고 자동 번역이 해당 컨텍스트를 사용하도록 Locadex/General Translation 플랫폼을 구성하는 방법을 설명합니다. 목표는 용어를 일관되게 사용하고, 제품명 및 기술 용어에 대해 올바른 “번역 제외” 동작을 보장하는 것입니다.

**구성 요소 위치:**

| 항목 | 위치 | 비고 |
|------|--------|------|
| **Glossary** (용어, 정의, 로케일별 번역) | Locadex 콘솔 → AI Context → Glossary | 용어를 일관되게 사용하도록 하고, 제품/기능 이름에 대한 “번역 제외”를 제어합니다. CSV로 일괄 업로드 가능. |
| **Locale Context** (언어별 지침) | Locadex 콘솔 → AI Context → Locale Context | 예: 한국어 – 알파벳과 한글 사이 띄어쓰기, 서식 규칙 등. |
| **Style Controls** (톤, 대상 독자, 프로젝트 설명) | Locadex 콘솔 → AI Context → Style Controls | 프로젝트 전역 설정으로, 모든 로케일에 적용됩니다. |
| **번역할 파일/로케일 지정** | Git → `gt.config.json` | `locales`, `defaultLocale`, `files`. 리포지토리에는 glossary나 프롬프트가 없습니다. |

정리하면: **자동 번역을 위한 컨텍스트는 Locadex 콘솔에서 설정합니다**(Glossary, Locale Context, Style Controls). **파일 및 로케일 설정은 Git에서 관리합니다**(`gt.config.json`). `gt.config.json`의 선택적 `dictionary` 키는 앱 UI 문자열(예: gt-next/gt-react)을 위한 것이며, 문서 MDX glossary용이 아닙니다. 문서 용어 관리는 콘솔에서 수행합니다.

<div id="context-and-constraints">
  ## 맥락과 제약 조건
</div>

<div id="legacy-tooling-wandb_docs_translation">
  ### 레거시 툴링 (wandb_docs_translation)
</div>

- **human_prompt.txt**: 절대 번역하면 안 되고(영어 그대로 유지해야 하는) W&B 제품/기능 이름을 나열합니다: Artifacts, Entities, Projects, Runs, Experiments, Datasets, Reports, Sweeps, Weave, Launch, Models, Teams, Users, Workspace, Registered Models. 링크/목록 컨텍스트(예: `[**word**](link)`)에서도 동일하게 적용됩니다.
- **system_prompt.txt**: 일반 규칙을 설명합니다(유효한 마크다운 유지, 코드 블록에서는 주석만 번역, 사전 사용, 링크 URL은 번역하지 않음; 일본어/한국어의 경우 알파벳과 CJK 문자를 전환할 때, 그리고 인라인 서식 주변에 공백 추가).
- **configs/language_dicts/ko.yaml**: 혼합 형태의 “번역 메모리”:
  - **영어 유지**(제품/기능 이름): 예: `Artifacts`, `Reports`, `Sweeps`, `Experiments`, `run`, `Weave expression`, `Data Visualization`, `Model Management`.
  - **한국어로 번역**: 예: `artifact` → 아티팩트, `sweep` → 스윕, `project` → 프로젝트, `workspace` → 워크스페이스, `user` → 사용자.

따라서 기존 규칙은 다음과 같았습니다. **제품/기능 이름(대문자로 시작하거나 UI/목록 컨텍스트에서 자주 사용됨)은 영어로 유지**하고, **일반 명사 용법**은 로케일용 사전을 따릅니다. Locadex 용어집은 각 로케일에 대해 “번역하지 않음”과 “X로 번역” 두 가지 정보를 모두 반영해야 합니다.

<div id="locadexgt-platform-behavior">
  ### Locadex/GT 플랫폼 동작 방식
</div>

- **Glossary**: 용어(원문 그대로) + 선택적 정의 + 선택적 로캘별 번역. “번역하지 않음”인 경우, 해당 로캘에서는 용어와 동일한 문자열을 번역 값으로 사용합니다(예: Term “W&amp;B”, Translation (ko) “W&amp;B”). “다음과 같이 번역”인 경우, Translation (ko)에 원하는 번역어를 설정합니다(예: “artifact” → “아티팩트”).
- **Locale Context**: 대상 로캘별 자유 형식의 지침(예: “라틴 문자와 한글 문자 사이에는 공백을 사용하세요”).
- **Style Controls**: 프로젝트 전체에 대한 하나의 세트(톤, 대상, 설명). 모든 로캘에 적용됩니다.
- AI Context를 변경해도 기존 번역 콘텐츠는 **자동으로** 재번역되지 않습니다. 이미 번역된 파일에 새 컨텍스트를 적용하려면 [Retranslate](https://generaltranslation.com/docs/platform/translations/retranslate)를 사용하세요.

<div id="step-by-step-process">
  ## 단계별 절차
</div>

<div id="1-gather-terminology-sources">
  ### 1. 용어 출처 수집
</div>

- **wandb_docs_translation에서 가져오기** (가능한 경우):
  - `configs/human_prompt.txt` → 절대 번역하면 안 되는 용어 목록.
  - `configs/language_dicts/ko.yaml` (그리고 이후 `ja.yaml`) → 용어 → 로케일 번역 매핑.
- **main 브랜치의 수동 번역본에서 가져오기** (선택): 일부 EN과 KO(또는 JA) 페이지를 비교해 제품명과 일반 용어가 어떻게 표현되었는지 확인한 뒤(예: “run” vs “실행”, “workspace” vs “워크스페이스”), 용어집 항목을 추가하거나 조정한다.

**에이전트 노트**: 에이전트가 외부 레포지토리를 읽을 수 없더라도, 이 레포지토리에 제공된 CSV와 로케일 컨텍스트 텍스트를 사람이 사용하면 런북을 그대로 따라갈 수 있다(아래의 런북과 선택적 CSV 참조).

<div id="2-build-or-obtain-a-glossary-csv">
  ### 2. 용어집 CSV를 만들거나 준비하기
</div>

- 이 저장소에 있는 한국어용 미리 준비된 용어집 CSV를 사용합니다: **runbooks/locadex-glossary-ko.csv**(아래 “Glossary CSV” 참조), 또는 다음을 포함하는 CSV를 직접 생성합니다:
  - **번역 금지 용어**: 용어 하나당 한 행; Definition은 선택 사항; `ko`(또는 “Translation (ko)”) = Term과 동일.
  - **번역된 용어**: 용어 하나당 한 행; Definition은 선택 사항; `ko` = 원하는 한국어 번역어.
- Locadex “Upload Context CSV”에서 요구하는 정확한 열 이름(`Term`, `Definition`, `ko` 또는 `Translation (ko)` 등)을 확인합니다. 콘솔에서 다른 이름을 기대하는 경우 CSV 헤더를 이에 맞게 수정합니다.
- **CSV 형식(정상 파싱을 위해)**: 파일이 올바르게 파싱되도록 표준 CSV 인용 규약을 사용합니다. 쉼표는 필드 구분자이며, 쉼표, 큰따옴표, 줄바꿈을 포함하는 필드는 반드시 큰따옴표로 감싸야 합니다. 큰따옴표로 감싼 필드 내부의 큰따옴표는 두 번(`""`) 이어서 써서 이스케이프합니다. 한 행에는 하나의 용어만 넣습니다(“run, Run”처럼 여러 변형을 한 셀에 넣지 마십시오). CSV를 프로그램을 통해 생성하거나 편집할 때는 CSV 라이브러리를 사용하거나 이런 필드를 명시적으로 인용해야 합니다. Term 또는 Definition에 따옴표로 감싸지지 않은 쉼표가 있으면 열 구분자로 처리되어 해당 행이 깨집니다.

<div id="3-configure-the-locadex-project-in-the-console">
  ### 3. 콘솔에서 Locadex 프로젝트 설정
</div>

1. [General Translation Dashboard](https://dash.generaltranslation.com/)에 로그인합니다.
2. wandb/docs 리포지토리에 연결된 프로젝트를 엽니다.
3. **AI Context**로 이동합니다(또는 이에 해당하는 메뉴: Glossary, Locale Context, Style Controls).

<div id="4-upload-or-add-glossary-terms">
  ### 4. 용어집 항목 업로드 또는 추가
</div>

- **옵션 A**: **Upload Context CSV**를 사용해 용어집(용어, 정의, 로케일 열)을 일괄로 가져옵니다. 플랫폼이 각 열을 용어집 항목과 로케일별 번역에 매핑합니다.
- **옵션 B**: 용어를 수동으로 추가합니다. Term, Definition(모델이 이해하는 데 도움을 줌), 그리고 한국어의 경우 번역(“번역하지 않음”이면 원어 그대로, “번역함”이면 한국어 문자열)을 입력합니다.

다음 사항은 반드시 충족해야 합니다:

- 반드시 영어로 유지해야 하는 제품/기능 이름: W&B, Weights & Biases, Artifacts, Runs, Experiments, Sweeps, Weave, Launch, Models, Reports, Datasets, Teams, Users, Workspace, Registered Models 등은 한국어 = 원문과 동일.
- 일관되게 번역해야 하는 용어: 예: artifact → 아티팩트, sweep → 스윕, project → 프로젝트, workspace → 워크스페이스, 그리고 `language_dicts/ko.yaml`(이후 `ja.yaml`)의 다른 항목들.

<div id="5-set-locale-context-for-korean">
  ### 5. 한국어 로케일 컨텍스트 설정
</div>

- 로케일 **ko** 를 선택합니다.
- 기존 system_prompt 와 한국어 문서 모범 사례를 반영하는 지침을 추가합니다. 예를 들어:
  - 라틴 문자와 한국어 문자(한글, 한자 포함) 사이를 전환할 때는 공백을 추가합니다.
  - 한국어 단어나 구의 일부에 인라인 서식(볼드, 이탤릭, 코드)을 사용할 때는, 마크다운이 올바르게 렌더링되도록 서식이 적용된 부분의 앞뒤에 공백을 추가합니다.
  - 코드 블록과 링크 URL 은 변경하지 말고, 필요한 경우에만 주변 설명 문장과 코드 내 주석만 번역합니다.

로케일 컨텍스트를 저장합니다.

<div id="6-set-style-controls-project-wide">
  ### 6. 스타일 설정 (프로젝트 전체 적용)
</div>

- **프로젝트 설명**: 예: “Weights & Biases (W&B) 문서: ML 실험 추적, 모델 레지스트리, LLM Ops를 위한 Weave 및 관련 제품.”
- **대상 독자**: 개발자와 ML 실무자.
- **톤**: 전문적이고 기술적이며 명료한 문체. 직역보다는 자연스럽게 읽히는 번역을 우선합니다.

저장합니다.

<div id="7-retranslate-if-needed">
  ### 7. 필요한 경우 재번역
</div>

- 이미 자동 번역된 콘텐츠가 있고 Glossary 또는 Locale Context를 변경했다면, 새 컨텍스트가 적용되도록 영향을 받는 파일에 대해 플랫폼의 **Retranslate** 기능을 사용하세요.

<div id="verification-and-testing">
  ## 검증 및 테스트
</div>

- **Glossary**: 업로드 후 Glossary 탭에서 일부 용어를 샘플로 점검합니다(번역 금지 항목과 번역된 항목 모두).
- **Locale Context**: 한국어(추후 일본어) 지침이 올바른 로케일에 저장되어 있는지 확인합니다.
- **Quality**: 샘플 페이지에 대해 번역을 실행하거나 트리거하여, 제품명이 영어로 유지되는지와 일반 용어가 Glossary와 일치하는지 확인합니다(예: 적절한 경우 artifact → 아티팩트).

<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-csv-upload-does-not-map-to-glossary">
  ### 문제: CSV 업로드가 용어집에 매핑되지 않음
</div>

- **원인**: 열 이름이 플랫폼에서 기대하는 이름과 일치하지 않을 수 있습니다.
- **해결 방법**: Locadex/GT 문서나 UI 내의 도움말에서 “Upload Context CSV” 열 이름(예: Term, Definition, locale code)을 확인하세요. CSV의 열 이름을 이에 맞게 변경한 후 다시 업로드하세요.

<div id="issue-terms-still-translated-when-they-should-stay-in-english">
  ### 문제: 영어로 유지해야 하는 용어가 여전히 번역됨
</div>

- **원인**: 용어가 Glossary에 없거나 “do not translate”가 설정되지 않음(해당 로캘의 번역이 누락되었거나 잘못됨).
- **해결 방법**: 대상 로캘에 대해 동일한 값으로 Glossary에 용어를 추가함(예: “Artifacts” → ko: “Artifacts”). 해당 용어가 제품/기능 이름임을 모델이 이해할 수 있도록 간단한 Definition을 추가함.

<div id="issue-japanese-or-another-locale-needs-different-rules">
  ### 문제: 일본어(또는 다른 로케일)에 다른 규칙이 필요함
</div>

- **원인**: 로케일별 선호 설정(예: 존댓말 사용, 띄어쓰기, 제품명에 가타카나 사용 등).
- **해결 방법**: 해당 로케일(예: ja)에 대해 별도의 Locale Context를 추가하고, 필요하다면 “ja” 열이 있는 추가 용어집 항목이나 일본어 수동 항목을 추가한다.

<div id="cleanup-instructions">
  ## 정리 지침
</div>

- 콘솔에서만 수행하는 설정의 경우 docs 리포지터리에서 임시 브랜치나 파일을 만들 필요는 없습니다.
- CSV를 생성하기 위해 일회성 스크립트를 만들었다면, 팀에서 해당 스크립트를 보관하기로 결정하지 않는 한 커밋하지 마세요(AGENTS.md와 일회성 스크립트에 대한 사용자 규칙을 참조하세요).

<div id="checklist">
  ## 체크리스트
</div>

- [ ] human_prompt, language_dicts/ko.yaml(필요하다면 ja도 포함)에서 용어를 수집했다.
- [ ] Glossary CSV를 생성하거나 확보했고, 업로드용 컬럼 이름을 확인했다.
- [ ] Locadex 콘솔에 로그인하고 올바른 프로젝트를 열었다.
- [ ] Glossary 용어(do-not-translate 및 번역 용어)를 업로드하거나 추가했다.
- [ ] 한국어용 Locale Context를 설정했다(추후 필요하다면 일본어도 설정).
- [ ] Style Controls(설명, 대상 독자, 톤)를 설정했다.
- [ ] 샘플 번역으로 검증하고, 필요 시 기존 콘텐츠를 재번역했다.

<div id="glossary-csv">
  ## 용어집 CSV
</div>

기본 한국어 용어집이 이 리포지토리에 제공됩니다: **runbooks/locadex-glossary-ko.csv**. 컬럼:

- **Term**: 문서에 나타나는 원본(영어) 용어.
- **Definition**: 짧은 설명(AI 보조용; 업로드 시 선택 사항).
- **ko**: 한국어 번역. “번역하지 않음”인 경우 Term과 동일한 문자열을 사용하고, “번역함”인 경우 원하는 한국어 문자열을 사용합니다.

`configs/language_dicts/ko.yaml`(또는 main 브랜치의 수동 KO 페이지)에서 더 많은 용어를 추가하려면, 동일한 컬럼 구조로 행을 이어서 추가하면 됩니다. Locadex 콘솔이 로케일 번역 컬럼 이름으로 다른 값을 요구하는 경우(예: “Translation (ko)”), 업로드 시 또는 업로드 전에 CSV에서 `ko` 컬럼 이름을 해당 이름으로 변경하세요.

<div id="csv-formatting-for-future-generation">
  ### 향후 생성을 위한 CSV 형식
</div>

용어집 CSV를 새로 만들거나(수동 또는 스크립트로) 내용을 추가할 때, 파일이 유효한 상태로 유지되도록 다음 규칙을 따르세요:

- **구분자**: 쉼표 (`,`). 필드 안에서는 쉼표를 사용하지 마세요. 반드시 사용해야 한다면 해당 필드를 큰따옴표로 감싸세요.
- **따옴표 처리**: 필드에 쉼표, 큰따옴표, 줄바꿈 중 하나라도 포함되면 그 필드를 큰따옴표(`"`)로 감싸세요. 일관성을 위해 모든 필드를 선택적으로 따옴표로 감싸도 됩니다.
- **이스케이프**: 큰따옴표로 감싼 필드 내부에서 실제 큰따옴표 문자를 나타내려면 큰따옴표 두 개(`""`)를 사용하세요.
- **행당 하나의 용어**: 각 행에는 하나의 용어만 담아야 합니다. 한 셀에 여러 변형을 나열하지 마세요 (예: Term 열에 “run, artifact”를 넣지 말고 “run”과 “artifact”를 각각 별도의 행으로 추가하세요).
- **도구**: CSV를 프로그램으로 생성할 때는 적절한 CSV 라이브러리를 사용하세요 (예: Python `csv` 모듈에서 `quoting=csv.QUOTE_MINIMAL` 또는 `QUOTE_NONNUMERIC` 사용). 이렇게 하면 Term 또는 Definition 안의 쉼표와 큰따옴표가 올바르게 처리됩니다.

<div id="notes">
  ## 노트
</div>

- **일본어는 나중에**: 일본어를 추가할 때는 `ja`에 대한 Locale Context를 다시 설정하세요(예: 공손한 말투 사용, 알파벳과 일본어 문자 사이의 공백, 인라인 서식 주변 공백 등). 그리고 `ja`용 용어집 항목을 추가하세요(동일한 방식: do-not-translate = 원문과 동일 유지, translate-as = 원하는 일본어 번역).
- **Git의 GT 설정**: `gt.config.json`에는 이미 `locales`와 `defaultLocale`이 있습니다. 용어집과 AI 컨텍스트는 그 파일에 저장되지 않고 콘솔에만 존재합니다.
- **참고 자료**: [GT Glossary](https://generaltranslation.com/docs/platform/ai-context/glossary), [Locale Context](https://generaltranslation.com/docs/platform/ai-context/locale-context), [Style Controls](https://generaltranslation.com/docs/platform/ai-context/style-controls), [Locadex for Mintlify](https://generaltranslation.com/docs/locadex/mintlify).