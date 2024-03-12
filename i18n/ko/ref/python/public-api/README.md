
# Import & Export API

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

## 클래스

[`class Api`](./api.md): wandb 서버를 쿼리하는 데 사용됩니다.

[`class File`](./file.md): File은 wandb에 의해 저장된 파일과 연관된 클래스입니다.

[`class Files`](./files.md): `File` 오브젝트의 반복 가능한 컬렉션입니다.

[`class Job`](./job.md)

[`class Project`](./project.md): 프로젝트는 실행을 위한 네임스페이스입니다.

[`class Projects`](./projects.md): `Project` 오브젝트의 반복 가능한 컬렉션입니다.

[`class QueuedRun`](./queuedrun.md): 엔티티와 프로젝트와 관련된 단일 대기 중인 실행입니다. 실행에 엑세스하려면 `run = queued_run.wait_until_running()` 또는 `run = queued_run.wait_until_finished()`를 호출합니다.

[`class Run`](./run.md): 엔티티와 프로젝트와 관련된 단일 실행입니다.

[`class RunQueue`](./runqueue.md)

[`class Runs`](./runs.md): 프로젝트와 선택적 필터와 관련된 실행의 반복 가능한 컬렉션입니다.

[`class Sweep`](./sweep.md): 스윕과 관련된 실행 세트입니다.