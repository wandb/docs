---
title: System metrics
description: wandb에 의해 자동으로 로그된 메트릭
displayed_sidebar: default
---

이 페이지는 W&B SDK가 추적하는 시스템 메트릭에 대한 자세한 정보를 제공합니다. 여기에는 특정 메트릭이 코드에서 어떻게 계산되는지도 포함되어 있습니다.

## CPU

### 프로세스 CPU 퍼센트 (CPU)
프로세스에 의한 CPU 사용량의 퍼센트로, 사용 가능한 CPU의 수로 정규화됩니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같은 공식을 통해 계산됩니다:

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

W&B는 이 메트릭에 `cpu` 태그를 할당합니다.

### CPU 퍼센트
시스템의 CPU 사용량을 각 코어별로 나타냅니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.cpu_percent(interval, percpu=True)
```

W&B는 이 메트릭에 `cpu.{i}.cpu_percent` 태그를 할당합니다.

### 프로세스 CPU 스레드 
프로세스에서 사용되는 스레드의 수를 나타냅니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.Process(pid).num_threads()
```

W&B는 이 메트릭에 `proc.cpu.threads` 태그를 할당합니다.

## Disk

기본적으로, 사용량 메트릭은 `/` 경로에 대해 수집됩니다. 모니터할 경로를 설정하려면 다음 설정을 사용하세요:

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### 디스크 사용량 퍼센트
지정된 경로에 대한 전체 시스템의 디스크 사용량을 퍼센트로 나타냅니다. 이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.disk_usage(path).percent
```
W&B는 이 메트릭에 `disk.{path}.usagePercen` 태그를 할당합니다.

### 디스크 사용량
지정된 경로에 대한 전체 시스템의 디스크 사용량을 기가바이트(GB)로 나타냅니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```
엑세스 가능한 경로가 샘플링되고, 각 경로의 디스크 사용량(GB)이 샘플에 추가됩니다.

W&B는 이 메트릭에 `disk.{path}.usageGB)` 태그를 할당합니다.

### 디스크 인
시스템의 전체 디스크 읽기를 메가바이트(MB) 단위로 나타냅니다. 이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

초기 디스크 읽기 바이트는 첫 샘플이 기록될 때 기록됩니다. 이후 샘플에서는 현재 읽기 바이트에서 초기 값을 뺀 차이를 계산합니다.

W&B는 이 메트릭에 `disk.in` 태그를 할당합니다.

### 디스크 아웃
시스템의 전체 디스크 쓰기를 메가바이트(MB) 단위로 나타냅니다. 이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

[Disk In](#disk-in)과 유사하게, 초기 디스크 쓰기 바이트는 첫 샘플이 기록될 때 기록됩니다. 이후 샘플에서는 현재 쓰기 바이트에서 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.out` 태그를 할당합니다.

## Memory

### 프로세스 메모리 RSS
프로세스의 메모리 레지던트 세트 크기(RSS)를 메가바이트(MB) 단위로 나타냅니다. RSS는 프로세스에 의해 점유된 메모리 중 메인 메모리(RAM)에 존재하는 부분을 의미합니다.

이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```
이는 프로세스의 RSS를 캡처하고 MB로 변환합니다.

W&B는 이 메트릭에 `proc.memory.rssMB` 태그를 할당합니다.

### 프로세스 메모리 퍼센트
총 사용 가능한 메모리의 퍼센트로 프로세스의 메모리 사용량을 나타냅니다.

이 메트릭은 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.Process(pid).memory_percent()
```

W&B는 이 메트릭에 `proc.memory.percent` 태그를 할당합니다.

### 메모리 퍼센트
총 사용 가능한 메모리의 퍼센트로 시스템의 전체 메모리 사용량을 나타냅니다.

이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.virtual_memory().percent
```

이는 전체 시스템의 총 메모리 사용량 퍼센트를 캡처합니다.

W&B는 이 메트릭에 `memory` 태그를 할당합니다.

### 메모리 사용 가능
시스템에서 사용 가능한 총 메모리를 메가바이트(MB) 단위로 나타냅니다.

이 메트릭은 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.virtual_memory().available / 1024 / 1024
```
이는 시스템에서 사용 가능한 메모리의 양을 가져와 MB로 변환합니다.

W&B는 이 메트릭에 `proc.memory.availableMB` 태그를 할당합니다.

## Network

### 네트워크 송신
네트워크를 통해 송신된 총 바이트 수를 나타냅니다.

이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```
초기 바이트 송신은 메트릭이 처음 초기화될 때 기록됩니다. 이후 샘플에서는 현재 바이트 송신에서 초기 값을 뺀 차이를 계산합니다.

W&B는 이 메트릭에 `network.sent` 태그를 할당합니다.

### 네트워크 수신

네트워크를 통해 수신된 총 바이트 수를 나타냅니다.

이 메트릭은 다음과 같은 공식으로 `psutil` 라이브러리를 사용하여 계산됩니다:

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```
[Network Sent](#network-sent)와 유사하게, 초기 바이트 수신은 메트릭이 처음 초기화될 때 기록됩니다. 이후 샘플에서는 현재 바이트 수신에서 초기 값을 뺀 차이를 계산합니다.

W&B는 이 메트릭에 `network.recv` 태그를 할당합니다.

## NVIDIA GPU

W&B는 NVIDIA GPU 메트릭을 캡처하기 위해 `pynvml` 라이브러리의 [수정된 버전](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py)을 사용합니다. 캡처된 메트릭에 대한 자세한 설명은 NVIDIA의 [가이드](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)를 참조하세요.

아래 설명된 메트릭 이외에도, 프로세스가 특정 GPU를 사용하는 경우 W&B는 해당 GPU의 메트릭을 `gpu.process.{gpu_index}...`로 캡처합니다.

W&B는 특정 GPU가 프로세스에 의해 사용되는지 확인하기 위해 다음 코드조각을 사용합니다:

```python
def gpu_in_use_by_this_process(gpu_handle: "GPUHandle", pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # base 프로세스를 찾을 수 없는 경우, 어떤 gpu 메트릭도 보고하지 않는다
        return False

    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)

    our_pids = {process.pid for process in our_processes}

    compute_pids = {
        process.pid
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)  # type: ignore
    }
    graphics_pids = {
        process.pid
        for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)  # type: ignore
    }

    pids_using_device = compute_pids | graphics_pids

    return len(pids_using_device & our_pids) > 0
```

### GPU 메모리 활용
각 GPU의 GPU 메모리 활용을 퍼센트로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memory` 태그를 할당합니다.

### GPU 메모리 할당
각 GPU의 총 사용 가능한 메모리 중 할당된 GPU 메모리를 퍼센트로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```
이는 각 GPU에 할당된 GPU 메모리의 퍼센트를 계산합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### GPU 메모리 할당 바이트
각 GPU에 할당된 GPU 메모리를 바이트 단위로 명시합니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocatedBytes` 태그를 할당합니다.

### GPU 활용
각 GPU의 GPU 활용을 퍼센트로 반영합니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### GPU 온도
각 GPU의 GPU 온도를 섭씨 단위로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### GPU 전력 사용량 와트
각 GPU의 GPU 전력 사용량을 와트 단위로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### GPU 전력 사용량 퍼센트

각 GPU의 전력 용량 중 GPU 전력 사용량을 퍼센트로 반영합니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

## AMD GPU
메트릭은 AMD에서 제공하는 `rocm-smi` 툴의 출력 (`stats`)로부터 추출됩니다 (`rocm-smi -a --json`).

### AMD GPU 활용
각 AMD GPU 디바이스에 대한 GPU 활용을 퍼센트로 나타냅니다.

```python
stats.get("GPU use (%)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### AMD GPU 메모리 할당
각 AMD GPU 디바이스에 대한 총 사용 가능한 메모리 중 GPU 메모리를 할당된 퍼센트를 나타냅니다.

```python
stats.get("GPU memory use (%)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### AMD GPU 온도
각 AMD GPU 디바이스에 대한 GPU 온도를 섭씨 단위로 나타냅니다.

```python
stats.get("Temperature (Sensor memory) (C)")
```
이는 각 AMD GPU의 온도를 가져옵니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### AMD GPU 전력 사용량 와트
각 AMD GPU 디바이스에 대한 GPU 전력 사용량을 와트 단위로 나타냅니다.

```python
stats.get("Average Graphics Package Power (W)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### AMD GPU 전력 사용량 퍼센트
각 AMD GPU 디바이스에 대한 전력 용량 중 GPU 전력 사용량을 퍼센트로 반영합니다.

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent`를 할당합니다.

## Apple ARM Mac GPU

### Apple GPU 활용
Apple GPU 디바이스, 특히 ARM Mac에서의 GPU 활용을 퍼센트로 나타냅니다.

이 메트릭은 `apple_gpu_stats` 바이너리에서 파생됩니다:
```python
raw_stats["utilization"]
```
W&B는 이 메트릭에 `gpu.0.gpu` 태그를 할당합니다.

### Apple GPU 메모리 할당
Apple GPU 디바이스, 특히 ARM Mac에서 사용 가능한 총 메모리 중 할당된 GPU 메모리를 퍼센트로 나타냅니다.

`apple_gpu_stats` 바이너리를 사용하여 추출됨:
```python
raw_stats["mem_used"]
```
이는 Apple GPU에 할당된 GPU 메모리의 퍼센트를 계산합니다.

W&B는 이 메트릭에 `gpu.0.memoryAllocated` 태그를 할당합니다.

### Apple GPU 온도
Apple GPU 디바이스, 특히 ARM Mac에서의 GPU 온도를 섭씨 단위로 표시합니다.

`apple_gpu_stats` 바이너리를 사용하여 파생됨:
```python
raw_stats["temperature"]
```

W&B는 이 메트릭에 `gpu.0.temp` 태그를 할당합니다.

### Apple GPU 전력 사용량 와트
Apple GPU 디바이스, 특히 ARM Mac에서의 GPU 전력 사용량을 와트 단위로 나타냅니다.

이 메트릭은 `apple_gpu_stats` 바이너리에서 얻습니다:
```python
raw_stats["power"]
```
이는 Apple GPU의 전력 사용량을 와트 단위로 계산합니다. 최대 전력 사용량은 16.5W로 하드 코딩되어 있습니다.

W&B는 이 메트릭에 `gpu.0.powerWatts` 태그를 할당합니다.

### Apple GPU 전력 사용량 퍼센트
Apple GPU 디바이스, 특히 ARM Mac에서의 전력 용량 중 GPU 전력 사용량을 퍼센트로 반영합니다.

`apple_gpu_stats` 바이너리를 사용하여 계산됨:
```python
(raw_stats["power"] / MAX_POWER_WATTS) * 100
```
이는 GPU의 전력 용량 대비 사용량을 퍼센트로 계산합니다. 최대 전력 사용량은 16.5W로 하드 코딩되어 있습니다.

W&B는 이 메트릭에 `gpu.0.powerPercent` 태그를 할당합니다.

## Graphcore IPU
Graphcore IPUs (Intelligence Processing Units)는 기계학습 작업을 위해 특별히 설계된 고유한 하드웨어 가속기입니다.

### IPU 디바이스 메트릭
이 메트릭은 특정 IPU 디바이스에 대한 다양한 통계를 나타냅니다. 각 메트릭은 식별하기 위해 디바이스 ID (`device_id`)와 메트릭 키 (`metric_key`)를 가집니다. W&B는 이 메트릭에 `ipu.{device_id}.{metric_key}` 태그를 할당합니다.

메트릭은 Graphcore의 `gcipuinfo` 바이너리와 상호작용하는 독점 `gcipuinfo` 라이브러리를 사용하여 추출됩니다. `sample` 메소드는 프로세스 ID (`pid`)와 관련된 각 IPU 디바이스에 대한 이러한 메트릭을 가져옵니다. 시간이 지남에 따라 변경되는 메트릭이나 디바이스 메트릭이 처음으로 가져올 때만 로그에 기록되며 중복되는 데이터를 로그하지 않으려 합니다.

각 메트릭에 대해, `parse_metric` 메소드를 사용해 원시 문자열 표현에서 메트릭의 값을 추출합니다. 그런 다음 메트릭은 여러 샘플에 걸쳐 `aggregate` 메소드를 사용하여 집계됩니다.

다음은 사용 가능한 메트릭과 그 단위의 목록입니다:

- **평균 보드 온도** (`average board temp (C)`): 섭씨 단위로 IPU 보드의 온도.
- **평균 다이 온도** (`average die temp (C)`): 섭씨 단위로 IPU 다이의 온도.
- **클록 속도** (`clock (MHz)`): IPU의 클록 속도(MHz).
- **IPU 전력** (`ipu power (W)`): IPU의 전력 소비(와트).
- **IPU 활용률** (`ipu utilisation (%)`): IPU 활용 비율.
- **IPU 세션 활용률** (`ipu utilisation (session) (%)`): 현재 세션에 특정된 IPU 활용 비율.
- **데이터 링크 속도** (`speed (GT/s)`): 초당 기가-전송 단위의 데이터 전송 속도.

## Google Cloud TPU

Tensor Processing Units (TPUs)은 Google에서 기계 학습 작업을 가속화하기 위해 커스터마이즈한 ASICs (Application Specific Integrated Circuits)입니다.

### TPU 활용률
이 메트릭은 Google Cloud TPU의 활용률을 퍼센트로 나타냅니다.

```python
tpu_name = os.environ.get("TPU_NAME")

compute_zone = os.environ.get("CLOUDSDK_COMPUTE_ZONE")
core_project = os.environ.get("CLOUDSDK_CORE_PROJECT")

from tensorflow.python.distribute.cluster_resolver import TPMClusterResolver

service_addr = TPUClusterResolver(
    [tpu_name], zone=compute_zone, project=core_project
).get_master()

service_addr = service_addr.replace("grpc://", "").replace(":8470", ":8466")

from tensorflow.python.profiler import profiler_client

result = profiler_client.monitor(service_addr, duration_ms=100, level=2)
```

W&B는 이 메트릭에 `tpu` 태그를 할당합니다.

## AWS Trainium

[AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)은 기계 학습 작업을 가속화하는 데 중점을 둔 AWS에서 제공하는 전문 하드웨어 플랫폼입니다. AWS의 `neuron-monitor` 툴을 이용하여 AWS Trainium 메트릭을 캡처합니다.

### Trainium Neuron Core Utilization
각 NeuronCore의 활용률을 퍼센트로 측정합니다. 코어별로 보고됩니다.

W&B는 이 메트릭에 `trn.{core_index}.neuroncore_utilization` 태그를 할당합니다.

### Trainium 호스트 메모리 사용량, 총계
호스트의 총 메모리 소비량을 바이트 단위로 나타냅니다.

W&B는 이 메트릭에 `trn.host_total_memory_usage` 태그를 할당합니다.

### Trainium Neuron 디바이스 총 메모리 사용량
Neuron 디바이스의 총 메모리 사용량을 바이트 단위로 나타냅니다.

W&B는 이 메트릭에 `trn.neuron_device_total_memory_usage)` 태그를 할당합니다.

### Trainium 호스트 메모리 사용량 세부사항:

호스트의 메모리 사용량에 대한 세부사항은 다음과 같습니다.

- **애플리케이션 메모리** (`trn.host_total_memory_usage.application_memory`): 애플리케이션에 의해 사용된 메모리.
- **상수** (`trn.host_total_memory_usage.constants`): 상수를 위한 메모리.
- **DMA 버퍼** (`trn.host_total_memory_usage.dma_buffers`): Direct Memory Access 버퍼를 위한 메모리.
- **텐서** (`trn.host_total_memory_usage.tensors`): 텐서를 위한 메모리.

### Trainium Neuron 코어 메모리 사용량 세부사항

각 NeuronCore에 대한 메모리 사용량 세부사항:

- **상수** (`trn.{core_index}.neuroncore_memory_usage.constants`)
- **모델 코드** (`trn.{core_index}.neuroncore_memory_usage.model_code`)
- **모델 공유 스크래치패드** (`trn.{core_index}.neuroncore_memory_usage.model_shared_scratchpad`)
- **런타임 메모리** (`trn.{core_index}.neuroncore_memory_usage.runtime_memory`)
- **텐서** (`trn.{core_index}.neuroncore_memory_usage.tensors`)

## OpenMetrics
OpenMetrics / Prometheus 호환 데이터와 사용자 정의 정규식 기반 메트릭 필터 지원을 제공하여 외부 엔드포인트로부터 메트릭을 캡처 및 기록합니다.

[NVIDIA DCGM-Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)를 사용하여 GPU 클러스터 성능을 모니터링하는 특정 사례에 대해 이 기능을 사용하는 방법의 자세한 예시는 [이 Report](https://wandb.ai/dimaduev/dcgm/reports/Monitoring-GPU-cluster-performance-with-NVIDIA-DCGM-Exporter-and-Weights-Biases--Vmlldzo0MDYxMTA1)를 참조하세요.