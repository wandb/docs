---
title: Import & Export API
menu:
  reference:
    identifier: ko-ref-python-public-api-_index
---

## 클래스

[`class Api`](./api.md): wandb 서버 를 쿼리하는 데 사용됩니다.

[`class File`](./file.md): File은 wandb에서 저장한 파일과 연결된 클래스입니다.

[`class Files`](./files.md): `File` 오브젝트 의 반복 가능한 컬렉션입니다.

[`class Job`](./job.md)

[`class Project`](./project.md): Project는 Runs의 네임스페이스입니다.

[`class Projects`](./projects.md): `Project` 오브젝트의 반복 가능한 컬렉션입니다.

[`class QueuedRun`](./queuedrun.md): 엔티티 및 프로젝트와 연결된 단일 대기열에 있는 Run입니다. `run = queued_run.wait_until_running()` 또는 `run = queued_run.wait_until_finished()`를 호출하여 Run에 엑세스합니다.

[`class Run`](./run.md): 엔티티 및 프로젝트와 연결된 단일 Run입니다.

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): 프로젝트와 연결된 Run의 반복 가능한 컬렉션이며, 선택적 필터입니다.

[`class Sweep`](./sweep.md): 스윕 과 연결된 Run의 집합입니다.
