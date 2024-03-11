---
description: Metrics automatically logged by wandb
displayed_sidebar: default
---

# 시스템 메트릭

이 페이지에서는 W&B SDK가 추적하는 시스템 메트릭에 대한 상세 정보를 제공하며, 특정 메트릭이 코드에서 어떻게 계산되는지에 대해서도 설명합니다.

## CPU

### 프로세스 CPU 퍼센트 (CPU)
프로세스에 의한 CPU 사용량의 백분율로, 사용 가능한 CPU 수로 정규화됩니다. 메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.Process(pid).cpu_percent() / psutil.cpu_count()
```

W&B는 이 메트릭에 `cpu` 태그를 할당합니다.

### CPU 퍼센트
코어별 시스템 CPU 사용량입니다. 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.cpu_percent(interval, percpu=True)
```

W&B는 이 메트릭에 `cpu.{i}.cpu_percent` 태그를 할당합니다.

### 프로세스 CPU 스레드 
프로세스가 사용하는 스레드 수입니다. 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.Process(pid).num_threads()
```

W&B는 이 메트릭에 `proc.cpu.threads` 태그를 할당합니다.

## 디스크

기본적으로, 사용량 메트릭은 `/` 경로에 대해 수집됩니다. 모니터링할 경로를 구성하려면 다음 설정을 사용하세요:

```python
run = wandb.init(
    settings=wandb.Settings(
        _stats_disk_paths=("/System/Volumes/Data", "/home", "/mnt/data"),
    ),
)
```

### 디스크 사용량 퍼센트
지정된 경로에 대한 시스템 디스크 사용량을 백분율로 나타냅니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.disk_usage(path).percent
```
W&B는 이 메트릭에 `disk.{path}.usagePercent` 태그를 할당합니다.

### 디스크 사용량
지정된 경로에 대한 시스템 디스크 사용량을 기가바이트(GB) 단위로 나타냅니다. 메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.disk_usage(path).used / 1024 / 1024 / 1024
```
접근 가능한 경로가 샘플링되고, 각 경로에 대한 디스크 사용량(GB 단위)이 샘플에 추가됩니다.


W&B는 이 메트릭에 `disk.{path}.usageGB` 태그를 할당합니다.

### 디스크 인
시스템 디스크 읽기 총량을 메가바이트(MB) 단위로 나타냅니다. 메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
(psutil.disk_io_counters().read_bytes - initial_read_bytes) / 1024 / 1024
```

초기 디스크 읽기 바이트는 첫 번째 샘플이 취해질 때 기록됩니다. 후속 샘플은 현재 읽기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.in` 태그를 할당합니다.

### 디스크 아웃
시스템 디스크 쓰기 총량을 메가바이트(MB) 단위로 나타냅니다. 이 메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
(psutil.disk_io_counters().write_bytes - initial_write_bytes) / 1024 / 1024
```

[디스크 인](#disk-in)과 마찬가지로, 초기 디스크 쓰기 바이트는 첫 번째 샘플이 취해질 때 기록됩니다. 후속 샘플은 현재 쓰기 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `disk.out` 태그를 할당합니다.

## 메모리

### 프로세스 메모리 RSS
프로세스의 메모리 거주 집합 크기(RSS)를 메가바이트(MB) 단위로 나타냅니다. RSS는 주 메모리(RAM)에 보관된 프로세스가 차지하는 메모리 부분입니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.Process(pid).memory_info().rss / 1024 / 1024
```
이는 프로세스의 RSS를 캡처하고 MB로 변환합니다.

W&B는 이 메트릭에 `proc.memory.rssMB` 태그를 할당합니다.

### 프로세스 메모리 퍼센트
프로세스의 메모리 사용량을 전체 사용 가능 메모리의 백분율로 나타냅니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.Process(pid).memory_percent()
```

W&B는 이 메트릭에 `proc.memory.percent` 태그를 할당합니다.

### 메모리 퍼센트
전체 시스템 메모리 사용량을 전체 사용 가능 메모리의 백분율로 나타냅니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.virtual_memory().percent
```

이는 전체 시스템의 총 메모리 사용량의 백분율을 캡처합니다.

W&B는 이 메트릭에 `memory` 태그를 할당합니다.

### 메모리 사용 가능
시스템의 전체 사용 가능 메모리를 메가바이트(MB) 단위로 나타냅니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음과 같이 계산됩니다:

```python
psutil.virtual_memory().available / 1024 / 1024
```
이는 시스템의 사용 가능 메모리 양을 검색하고 MB로 변환합니다.

W&B는 이 메트릭에 `proc.memory.availableMB` 태그를 할당합니다.

## 네트워크

### 네트워크 전송
네트워크를 통해 전송된 총 바이트를 나타냅니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.net_io_counters().bytes_sent - initial_bytes_sent
```
초기에 전송된 바이트는 메트릭이 처음 초기화될 때 기록됩니다. 후속 샘플은 현재 전송된 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.sent` 태그를 할당합니다.

### 네트워크 수신

네트워크를 통해 수신된 총 바이트를 나타냅니다.

메트릭은 `psutil` 라이브러리를 사용하여 다음 공식으로 계산됩니다:

```python
psutil.net_io_counters().bytes_recv - initial_bytes_received
```
[네트워크 전송](#network-sent)과 마찬가지로, 초기에 수신된 바이트는 메트릭이 처음 초기화될 때 기록됩니다. 후속 샘플은 현재 수신된 바이트와 초기 값의 차이를 계산합니다.

W&B는 이 메트릭에 `network.recv` 태그를 할당합니다.

## NVIDIA GPU

W&B는 `pynvml` 라이브러리의 [적응된 버전](https://github.com/wandb/wandb/blob/main/wandb/vendor/pynvml/pynvml.py)을 사용하여 NVIDIA GPU 메트릭을 캡처합니다. NVIDIA가 제공하는 [이 가이드](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)에서 캡처된 메트릭에 대한 자세한 설명을 참조하세요.

아래에 설명된 메트릭 외에도, 프로세스가 특정 GPU를 사용하는 경우, W&B는 해당 메트릭을 `gpu.process.{gpu_index}...`로 캡처합니다.

W&B는 프로세스가 특정 GPU를 사용하는지 확인하기 위해 다음 코드조각을 사용합니다:

```python
def gpu_in_use_by_this_process(gpu_handle: "GPUHandle", pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # 기본 프로세스를 찾을 수 없는 경우 GPU 메트릭을 보고하지 않음
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

### GPU 메모리 사용률
각 GPU에 대한 GPU 메모리 사용률을 백분율로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).memory
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memory` 태그를 할당합니다.

### GPU 메모리 할당
각 GPU에 대해 전체 사용 가능 메모리의 백분율로 할당된 GPU 메모리를 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used / memory_info.total * 100
```
이는 각 GPU에 할당된 GPU 메모리의 백분율을 계산합니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### GPU 메모리 할당 바이트
각 GPU에 할당된 GPU 메모리를 바이트 단위로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
memory_info.used
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocatedBytes` 태그를 할당합니다.

### GPU 사용률
각 GPU에 대한 GPU 사용률을 백분율로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
```

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### GPU 온도
각 GPU의 GPU 온도를 섭씨로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
```

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### GPU 전력 사용 와트
각 GPU의 GPU 전력 사용량을 와트로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### GPU 전력 사용 퍼센트

각 GPU의 전력 사용량을 전력 용량의 백분율로 나타냅니다.

```python
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
(power_watts / power_capacity_watts) * 100
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

## AMD GPU
메트릭은 AMD가 제공하는 `rocm-smi` 툴(`rocm-smi -a --json`)의 출력(`stats`)에서 추출됩니다.

### AMD GPU 사용률
각 AMD GPU 장치에 대한 GPU 사용률을 백분율로 나타냅니다.

```python
stats.get("GPU use (%)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.gpu` 태그를 할당합니다.

### AMD GPU 메모리 할당
각 AMD GPU 장치에 대해 전체 사용 가능 메모리의 백분율로 할당된 GPU 메모리를 나타냅니다.

```python
stats.get("GPU memory use (%)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.memoryAllocated` 태그를 할당합니다.

### AMD GPU 온도
각 AMD GPU 장치의 GPU 온도를 섭씨로 나타냅니다.

```python
stats.get("Temperature (Sensor memory) (C)")
```
이는 각 AMD GPU의 온도를 가져옵니다.

W&B는 이 메트릭에 `gpu.{gpu_index}.temp` 태그를 할당합니다.

### AMD GPU 전력 사용 와트
각 AMD GPU 장치의 GPU 전력 사용량을 와트로 나타냅니다.

```python
stats.get("Average Graphics Package Power (W)")
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerWatts` 태그를 할당합니다.

### AMD GPU 전력 사용 퍼센트
각 AMD GPU 장치의 전력 사용량을 전력 용량의 백분율로 나타냅니다.

```python
(
    stats.get("Average Graphics Package Power (W)")
    / float(stats.get("Max Graphics Package Power (W)"))
    * 100
)
```

W&B는 이 메트릭에 `gpu.{gpu_index}.powerPercent` 태그를 할당합니다.

## Apple ARM Mac GPU

### Apple GPU 사용률
특히 ARM Mac에서 Apple GPU 장치의 GPU 사용률을 백분율로 나타냅니다.

메트릭은 `apple_gpu_stats` 바이너리에서 파생됩니다:
```python
raw_stats["utilization"]
```
W&B는 이 메트릭에 `gpu.0.gpu` 태그를 할당합니다.

### Apple GPU 메모리 할당
ARM Mac에서 Apple GPU 장치에 대해 전체 사용 가능 메모리의 백분율로 할당된 GPU 메모리를 나타냅니다.

`apple_gpu_stats` 바이너리를 사용하여 추출됩니다:
```python
raw_stats["mem_used"]
```
이는 Apple GPU에 할당된 GPU 메모리의 백분율을 계산합니다.

W&B는 이 메트릭에 `gpu.0.memoryAllocated` 태그를 할당합니다.

### Apple GPU 온도
ARM Mac에서 Apple GPU 장치의 GPU 온도를 섭