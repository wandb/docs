# Write-ahead log

The write-ahead log (WAL) is an optional feature that improves the resiliency and robustness of API writes when using the Weave client. Instead of holding outgoing requests in memory, the WAL writes them to disk first, ensuring that data is not lost if the client process crashes, the server is unreachable, or a pod runs out of memory before data is flushed.

## Overview

By default, the Weave client buffers outgoing API requests in memory and sends them to the server. If something goes wrong before the flush completes—such as a server outage, network interruption, or out-of-memory event—that buffered data is lost permanently.

When WAL is enabled, every API request is written to a local [JSONL](https://jsonlines.org/) file on disk before it is sent. The client then flushes those records to the server in the background. If a flush fails, the records remain on disk and are retried on the next run.

## Prerequisites

- Weave client version `X.X.X` or later *(WAL ships in the next client release)*
- Write access to the `.weave/wal/` directory in your working environment

## Enable WAL

WAL is opt-in. To enable it, set the following environment variable before running your script:

```bash
export WEAVE_ENABLE_WAL=true
```

Or set it inline for a single run:

```bash
WEAVE_ENABLE_WAL=true python experiment.py
```

> **Note:** In a future Weave release, WAL will be enabled by default.

## How it works

When WAL is enabled and your script runs, the client writes each pending API request to a JSONL file under:

```
.weave/wal/<entity>/<project>/
```

Each parallel process gets its own JSONL file. A file contains the serialized request payloads that would otherwise be sent directly to the server, such as object creates, file creates, and call start/end events.

Once the client is ready to flush, it reads from those files and sends the requests to the server. On success, the files are deleted. If you navigate to your W&B project before flushing, no data appears—data is only visible in the UI after a successful flush.

## Inspect WAL files

You can inspect the contents of the WAL directory at any time to see what is queued for upload:

```bash
ls .weave/wal/<entity>/<project>/
```

To view the raw request payloads in a file:

```bash
cat .weave/wal/<entity>/<project>/<file>.jsonl
```

Each line in the file is a JSON object representing a single API request payload.

## Environment variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `WEAVE_ENABLE_WAL` | `bool` | `false` | Enables the write-ahead log. Set to `true` to activate. |
| `WEAVE_DISABLE_WAL_SENDER` | `bool` | `false` | Disables background flushing of WAL files. When set to `true`, requests are written to disk but not sent to the server. Useful for testing and inspecting WAL behavior. |

## Example

The following example shows a basic Weave script run in four parallel processes with WAL enabled.

**`experiment.py`**

```python
import weave

weave.init("my-project")

class Thing(weave.Object):
    value: str

@weave.op()
def my_func(thing: Thing) -> str:
    return f"processed {thing.value}"

thing = Thing(value="a")
my_func(thing)
```

**Run with WAL enabled:**

```bash
WEAVE_ENABLE_WAL=true python process.py  # spawns 4 parallel experiment.py processes
```

After the run, WAL files are written to `.weave/wal/<entity>/<project>/`. On the next run with `WEAVE_DISABLE_WAL_SENDER` unset (or set to `false`), the client flushes the existing WAL files and any new requests to the server. Runs appear in the W&B project UI after a successful flush.

## Failure recovery

If the server is unavailable during a run, WAL files accumulate on disk. The next time the client runs with flushing enabled, it picks up those files and retries the upload. No manual intervention is required.

> **Caution:** If the `.weave/wal/` directory is deleted before flushing, the queued data is permanently lost.

## Limitations

- WAL requires a writable local filesystem. It is not suitable for read-only or ephemeral environments where the working directory is not persisted across runs.
- WAL files are process-local. If you move the working directory between runs, point the new environment to the same `.weave/wal/` path to ensure pending files are flushed.

## Related resources

- [Weave quickstart](https://wandb.github.io/weave/)
- [Weave client configuration](https://wandb.github.io/weave/)
- [W&B runs and experiments](https://docs.wandb.ai/guides/runs)
