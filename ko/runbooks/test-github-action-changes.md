---
title: GitHub Actions 변경 사항 테스트
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # 에이전트용 프롬프트: wandb/docs에서 GitHub Actions 변경 내용 테스트
</div>

<div id="requirements">
  ## 요구 사항
</div>

- **W&B 직원 액세스**: 내부 W&B 시스템에 접근할 수 있는 W&B 직원이어야 합니다.
- **GitHub 포크**: 워크플로 변경 사항을 테스트하기 위한 wandb/docs의 개인 포크가 필요합니다. 이 포크에서 기본 브랜치에 푸시할 수 있는 권한과 브랜치 보호 규칙을 건너뛸 수 있는 권한이 있어야 합니다.

<div id="agent-prerequisites">
  ## 에이전트 사전 요구 사항
</div>

시작하기 전에 다음 정보를 준비하세요:

1. **GitHub 사용자명** - 먼저 `git remote -v`로 포크 원격 저장소를 확인한 다음, `git config`에서 사용자명을 확인하세요. 두 곳 모두에서 찾을 수 없을 때만 사용자에게 물어보세요.
2. **포크 상태** - wandb/docs 저장소에 대한 포크가 있고, 기본 브랜치에 푸시할 수 있으며 브랜치 보호를 우회할 수 있는 권한이 있는지 확인하세요.
3. **테스트 범위** - 어떤 구체적인 변경 사항(의존성 업그레이드, 기능 변경 등)을 테스트하고 있는지 질문하세요.

<div id="task-overview">
  ## 작업 개요
</div>

wandb/docs 리포지토리에서 GitHub Actions 워크플로우의 변경 내용을 테스트합니다.

<div id="context-and-constraints">
  ## 맥락 및 제약 조건
</div>

<div id="repository-setup">
  ### 저장소 설정
</div>

- **메인 저장소**: `wandb/docs` (origin)
- **테스트용 포크**: `<username>/docs` (fork remote) - `git remoter -v`로 명확하지 않다면, 사용자에게 포크의 엔드포인트를 물어보세요.
- **중요**: PR에서 GitHub Actions는 항상 PR 브랜치가 아니라 기준(base) 브랜치(main)에서 실행됩니다.
- **Mintlify 배포 제한 사항**: Mintlify 배포와 `link-rot` 검사는 포크가 아니라 메인 wandb/docs 저장소에 대해서만 빌드를 수행합니다. 포크에서는 `validate-mdx` GitHub Action이 포크 PR에서 `mint dev`와 `mint broken-links` 명령의 상태만 확인합니다.

**에이전트 노트**: 다음 작업을 수행해야 합니다:

1. 기존 포크용 원격 저장소를 확인하기 위해 `git remote -v`를 실행하고, 가능하다면 URL에서 username을 추출하세요.
2. remotes에서 username을 찾지 못한 경우, `git config`에서 GitHub username을 확인하세요.
3. 두 곳 모두에서 찾지 못한 경우에만 사용자에게 GitHub username을 물어보세요.
4. 테스트에 사용할 수 있는 wandb/docs 포크가 있는지 확인하세요.
5. 포크에 직접 push할 수 없다면, 사용자가 그 브랜치를 기반으로 push할 수 있도록 wandb/docs에 임시 브랜치를 생성하세요.

<div id="testing-requirements">
  ### 테스트 요구 사항
</div>

워크플로 변경 사항을 테스트하려면 다음을 수행해야 합니다:

1. 포크의 `main`을 메인 저장소의 `main`과 동기화하면서, 모든 임시 커밋을 제거합니다.
2. 변경 사항을 포크의 메인 브랜치(별도의 기능 브랜치만이 아니라)에 적용합니다.
3. 워크플로를 트리거하기 위해 내용 변경을 포함한 테스트용 PR을 포크의 `main`을 대상으로 생성합니다.

<div id="step-by-step-testing-process">
  ## 단계별 테스트 절차
</div>

<div id="1-initial-setup">
  ### 1. 초기 설정
</div>

```bash
# 기존 원격 저장소 확인
git remote -v

# fork 원격 저장소가 있으면 fork URL에서 사용자 이름 확인
# fork 원격 저장소가 없으면 git config에서 사용자 이름 확인
git config user.name  # or git config github.user

# 원격 저장소나 config에서 찾을 수 없는 경우에만 사용자에게 GitHub 사용자 이름 또는 fork 정보 요청
# 질문 예시: "테스트에 사용할 fork의 GitHub 사용자 이름이 무엇인가요?"

# fork 원격 저장소가 없으면 추가:
git remote add fork https://github.com/<username>/docs.git  # <username>을 실제 사용자 이름으로 변경
```


<div id="2-sync-fork-and-prepare-test-branch">
  ### 2. 포크 동기화 및 테스트 브랜치 준비
</div>

```bash
# origin에서 최신 내용 가져오기
git fetch origin

# main 체크아웃 후 origin/main으로 하드 리셋하여 깨끗한 동기화 보장
git checkout main
git reset --hard origin/main

# fork에 강제 푸시하여 동기화 (fork의 임시 커밋 모두 삭제)
git push fork main --force

# 워크플로우 변경 사항을 위한 테스트 브랜치 생성
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. 워크플로 변경 사항 적용
</div>

워크플로 파일을 수정합니다. 종속성을 업그레이드하는 경우:

- `uses:` 구문에서 버전 번호를 업데이트합니다.
- 해당 종속성이 여러 곳에서 사용된다면 두 워크플로 파일을 모두 확인합니다.

**프로 팁**: 어떤 런북이든 마무리하기 전에, 다음과 같은 프롬프트로 AI 에이전트에게 검토를 요청해 보세요.

> "Please review this runbook and suggest improvements to make it more useful for AI agents. Focus on clarity, completeness, and removing ambiguity."

<div id="5-commit-and-push-to-forks-main">
  ### 5. 포크한 저장소의 main 브랜치에 커밋하고 푸시하기
</div>

```bash
# 모든 변경 사항 커밋
git add -A
git commit -m "test: [Description of what you're testing]"

# 포크의 main 브랜치에 푸시
git push fork HEAD:main --force-with-lease
```

**포크 접근을 위한 에이전트 지침**:
포크에 직접 푸시할 수 없는 경우:

1. 변경 사항이 포함된 임시 브랜치를 wandb/docs에 생성합니다.
2. 사용자에게 다음 명령을 실행하라고 안내합니다:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 다음 주소에서 PR을 생성하도록 안내합니다: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. 테스트가 끝나면 wandb/docs에서 임시 브랜치를 삭제하는 것을 잊지 마세요.


<div id="6-create-test-pr">
  ### 6. 테스트 PR 만들기
</div>

```bash
# 업데이트된 포크 main에서 새 브랜치 생성
git checkout -b test-pr-[description]

# 워크플로우를 트리거할 소규모 콘텐츠 변경
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# 커밋 및 푸시
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

그런 다음 GitHub UI에서 `<username>:test-pr-[description]` 브랜치에서 `<username>:main` 브랜치로 향하는 PR을 생성하세요.


<div id="7-monitor-and-verify">
  ### 7. 모니터링 및 검증
</div>

예상 동작:

1. GitHub Actions 봇이 "Generating preview links..."라는 초기 댓글을 생성한다
2. 워크플로가 오류 없이 완료되어야 한다

다음을 확인하세요:

- ✅ 워크플로가 성공적으로 완료됨
- ✅ 미리보기 댓글이 생성되고 업데이트됨
- ✅ 링크가 지정한 override URL을 사용함
- ✅ 파일 분류가 정상 동작함 (Added/Modified/Deleted/Renamed)
- ❌ Actions 로그에 어떤 오류도 없어야 함
- ❌ 보안 경고나 시크릿 노출이 없어야 함

<div id="8-cleanup">
  ### 8. 정리
</div>

테스트를 마친 후:

```bash
# 포크의 main을 upstream과 일치하도록 초기화
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# 포크와 origin에서 테스트 브랜치 삭제
git branch -D test-[description]-[date] test-pr-[description]
```


<div id="common-issues-and-solutions">
  ## 자주 발생하는 문제와 해결 방법
</div>

<div id="issue-permission-denied-when-pushing-to-fork">
  ### 문제: 포크에 푸시할 때 권한이 거부됨
</div>

- GitHub 토큰이 읽기 전용(read-only)일 수 있습니다
- 해결 방법: SSH를 사용하거나 로컬 머신에서 직접 푸시하세요

<div id="issue-workflows-not-triggering">
  ### 문제: 워크플로가 실행되지 않음
</div>

- 참고: 워크플로는 PR 브랜치가 아니라 기본 브랜치(main)에서 실행됩니다
- 변경 사항이 포크의 기본 브랜치(main)에 있는지 확인하세요

<div id="issue-changed-files-not-detected">
  ### 문제: 변경된 파일이 감지되지 않음
</div>

- 콘텐츠 변경 사항이 추적 대상 디렉터리(content/, static/, assets/ 등)에 있는지 확인하세요.
- 워크플로 설정에서 `files:` 필터를 확인하세요.

<div id="testing-checklist">
  ## 테스트 체크리스트
</div>

- [ ] 사용자에게 GitHub 사용자명과 fork 정보를 요청함
- [ ] 두 원격(origin, fork)이 모두 설정되어 있음
- [ ] 워크플로(workflow) 변경 사항을 관련된 두 파일 모두에 적용함
- [ ] 변경 사항을 fork의 기본 브랜치(main)에 푸시함(직접 또는 사용자를 통해)
- [ ] 내용 변경이 포함된 테스트 PR을 생성함
- [ ] 미리보기(preview) 댓글이 성공적으로 생성됨
- [ ] GitHub Actions 로그에 오류가 없음
- [ ] 테스트 후 fork의 기본 브랜치(main)를 원래대로 되돌림(reset)
- [ ] 임시 브랜치를 wandb/docs에서 정리함(생성된 경우)