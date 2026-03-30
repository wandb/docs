---
title: GitHub Action 변경 테스트
---

<div id="agent-prompt-testing-github-actions-changes-in-wandbdocs">
  # Agent 프롬프트: wandb/docs의 GitHub Actions 변경 사항 테스트
</div>

<div id="requirements">
  ## 요구 사항
</div>

* **W&amp;B 직원 접근 권한**: 내부 W&amp;B 시스템에 접근할 수 있는 W&amp;B 직원이어야 합니다.
* **GitHub 포크**: 워크플로 변경 사항을 테스트하기 위한 wandb/docs의 개인 포크가 필요합니다. 이 포크에서는 기본 브랜치에 푸시하고 브랜치 보호 규칙을 우회할 수 있는 권한이 있어야 합니다.

<div id="agent-prerequisites">
  ## Agent 사전 요구 사항
</div>

시작하기 전에 다음 정보를 수집하세요:

1. **GitHub username** - 먼저 `git remote -v`로 포크 원격 저장소가 있는지 확인한 다음, `git config`에서 사용자 이름을 확인하세요. 두 위치 모두에서 찾을 수 없는 경우에만 사용자에게 물어보세요.
2. **포크 status** - 기본 브랜치에 푸시하고 브랜치 보호를 우회할 권한이 있는 wandb/docs 포크를 가지고 있는지 확인하세요.
3. **Test scope** - 어떤 구체적인 변경 사항을 테스트하는지 물어보세요(의존성 업그레이드, 기능 변경 등).

<div id="task-overview">
  ## 작업 Overview
</div>

wandb/docs 저장소에서 GitHub Actions 워크플로의 변경 사항을 테스트합니다.

<div id="context-and-constraints">
  ## 배경 및 제약 사항
</div>

<div id="repository-setup">
  ### 저장소 설정
</div>

* **메인 저장소**: `wandb/docs` (origin)
* **테스트용 포크**: `<username>/docs` (포크 원격 저장소) - `git remoter -v`만으로 명확하지 않으면 사용자에게 포크의 엔드포인트를 물어보세요.
* **중요**: PR의 GitHub Actions는 항상 PR 브랜치가 아니라 베이스 브랜치(main)에서 실행됩니다.
* **Mintlify 배포 제한 사항**: Mintlify 배포와 `link-rot` 확인은 포크가 아니라 메인 `wandb/docs` 저장소에서만 빌드됩니다. 포크에서는 포크 PR의 `validate-mdx` GitHub Action이 `mint dev` 및 `mint broken-links` 명령어의 status를 확인합니다.

**에이전트 참고**: 다음을 수행해야 합니다.

1. 기존 포크 원격 저장소가 있는지 `git remote -v`로 확인하고, 있으면 URL에서 사용자 이름을 추출합니다.
2. 원격 저장소에서 사용자 이름을 찾지 못하면 `git config`에서 GitHub 사용자 이름을 확인합니다.
3. 두 위치 모두에서 찾지 못한 경우에만 사용자에게 GitHub 사용자 이름을 물어봅니다.
4. 테스트에 사용할 수 있는 `wandb/docs` 포크가 있는지 확인합니다.
5. 포크에 직접 푸시할 수 없으면 사용자가 푸시할 수 있도록 `wandb/docs`에 임시 브랜치를 만듭니다.

<div id="testing-requirements">
  ### 테스트 요구 사항
</div>

워크플로 변경 사항을 테스트하려면 다음을 수행해야 합니다.

1. 포크의 `main`을 메인 리포지토리의 `main`과 동기화하고, 임시 커밋은 모두 버립니다.
2. 변경 사항을 포크의 메인 브랜치에 적용합니다(기능 브랜치에만 적용하지 말 것).
3. 콘텐츠를 변경해 워크플로가 트리거되도록 한 뒤, 포크의 `main`을 대상으로 테스트 PR을 생성합니다.

<div id="step-by-step-testing-process">
  ## step별 테스트 절차
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

# main 체크아웃 후 origin/main으로 하드 재설정하여 깨끗한 동기화 보장
git checkout main
git reset --hard origin/main

# fork에 강제 푸시하여 동기화 (fork의 임시 커밋 모두 삭제)
git push fork main --force

# 워크플로 변경 사항을 위한 테스트 브랜치 생성
git checkout -b test-[description]-[date]
```


<div id="3-apply-workflow-changes">
  ### 3. 워크플로 변경 사항 적용
</div>

워크플로 파일을 수정하세요. 종속성 업그레이드의 경우:

* `uses:` 구문의 버전 번호를 업데이트합니다
* 종속성이 여러 위치에서 사용되는 경우 두 워크플로 파일을 모두 확인합니다

**유용한 팁**: 런북을 최종 확정하기 전에 다음과 같은 프롬프트로 AI agent에게 검토를 요청하세요:

> &quot;이 런북을 검토하고 AI agent에 더 유용하도록 개선 사항을 제안해 주세요. 명확성, 완전성, 그리고 모호성 제거에 중점을 두세요.&quot;

<div id="5-commit-and-push-to-forks-main">
  ### 5. 포크의 main 브랜치에 커밋하고 푸시
</div>

```bash
# 모든 변경 사항 커밋
git add -A
git commit -m "test: [Description of what you're testing]"

# 포크의 main 브랜치에 푸시
git push fork HEAD:main --force-with-lease
```

**포크 액세스를 위한 Agent 지침**:
포크에 직접 푸시할 수 없는 경우:

1. 변경 사항이 포함된 임시 브랜치를 wandb/docs에 생성하세요
2. 사용자에게 다음 명령어를 제공하세요:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. 다음에서 PR을 생성하도록 안내하세요: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. 테스트가 끝나면 wandb/docs에서 임시 브랜치를 삭제하는 것을 잊지 마세요


<div id="6-create-test-pr">
  ### 6. 테스트용 PR 만들기
</div>

```bash
# 업데이트된 포크 main에서 새 브랜치 생성
git checkout -b test-pr-[description]

# 워크플로를 트리거할 소규모 콘텐츠 변경
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# 커밋 및 푸시
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

그런 다음 GitHub UI에서 `<username>:test-pr-[description]` 브랜치에서 `<username>:main` 브랜치로 PR을 생성합니다


<div id="7-monitor-and-verify">
  ### 7. 모니터링 및 확인
</div>

예상 동작:

1. GitHub Actions 봇이 &quot;Generating preview links...&quot;라는 초기 댓글을 생성합니다
2. 워크플로가 오류 없이 완료되어야 합니다

다음 사항을 확인하세요:

* ✅ 워크플로가 성공적으로 완료됨
* ✅ 프리뷰 댓글이 생성되고 업데이트됨
* ✅ 링크가 override URL을 사용함
* ✅ 파일 분류가 올바르게 작동함 (Added/Modified/Deleted/Renamed)
* ❌ Actions 로그에 오류가 없음
* ❌ 보안 경고 또는 노출된 시크릿이 없음

<div id="8-cleanup">
  ### 8. 정리
</div>

테스트 후:

```bash
# 포크의 main을 upstream과 일치하도록 재설정
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

* GitHub 토큰이 읽기 전용일 수 있습니다
* 해결 방법: SSH를 사용하거나 로컬 컴퓨터에서 직접 푸시합니다

<div id="issue-workflows-not-triggering">
  ### 문제: 워크플로가 트리거되지 않음
</div>

* 참고: 워크플로는 PR 브랜치가 아니라 기본 브랜치(main)에서 실행됩니다
* 변경 사항이 포크의 main 브랜치에 반영되어 있는지 확인하세요

<div id="issue-changed-files-not-detected">
  ### 문제: Changed 파일이 감지되지 않음
</div>

* 콘텐츠 변경 사항이 추적되는 디렉터리(content/, static/, assets/ 등)에 있는지 확인합니다.
* 워크플로 설정의 `files:` 필터를 확인합니다.

<div id="testing-checklist">
  ## 테스트 체크리스트
</div>

* [ ] 사용자에게 GitHub 사용자 이름과 포크 정보를 요청함
* [ ] 두 원격 저장소(`origin` 및 포크)가 모두 구성되어 있음
* [ ] 워크플로 변경 사항이 관련 파일 두 개 모두에 적용됨
* [ ] 변경 사항이 포크의 main 브랜치에 푸시됨(직접 또는 사용자를 통해)
* [ ] 콘텐츠 변경 사항이 포함된 테스트 PR이 생성됨
* [ ] 미리보기 댓글이 성공적으로 생성됨
* [ ] GitHub Actions 로그에 오류가 없음
* [ ] 테스트 후 포크의 main 브랜치가 재설정됨
* [ ] wandb/docs의 임시 브랜치가 정리됨(생성된 경우)