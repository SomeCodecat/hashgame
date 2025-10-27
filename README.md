# Rust Miner

High-performance SHA-256 mining with CPU and GPU support.

## Usage

### Quick Start

```bash
# CPU mining (uses all cores by default)
./target/release/rust_miner --name YOUR_NAME

# GPU mining (recommended for best performance)
./target/release/rust_miner --name YOUR_NAME --gpu
```

## Best Performance Commands

### Maximum GPU Performance

```bash
# Optimal for most GPUs (1M hashes per batch, 2 workers)
./target/release/rust_miner --name AS --gpu --gpu-workers 2

# High-end GPUs (larger batches, more workers)
GPU_BATCH_SIZE=4194304 ./target/release/rust_miner --name AS --gpu --gpu-workers 3

# Fine-tune work-group size if needed
./target/release/rust_miner --name AS --gpu --work-group-size 256
```

**Expected performance:** ~1000-1100 MH/s on RTX 4070 SUPER

### Maximum CPU Performance

```bash
# Use all CPU cores (optimal for most systems)
./target/release/rust_miner --name AS

# Explicit worker count with optimizations
./target/release/rust_miner --name AS --workers 12 --no-verify --short-secs 5

# Use only physical cores (sometimes faster than hyperthreading)
./target/release/rust_miner --name AS --workers 6
```

**Expected performance:** ~40-80 MH/s on modern CPUs

## Command Reference

### Required

- `--name PREFIX` — Miner name prefix (suffix '-B6' appended automatically)
  - Can also use `NAME_PREFIX` environment variable

### Mining Options

- `--gpu` — Enable OpenCL GPU mining
- `--workers N`, `-w N` — Number of CPU worker threads (default: all CPUs)
- `--gpu-workers N` — Number of GPU worker threads (default: 2)

### Performance Tuning

- `--work-group-size N` — OpenCL local work-group size (default: auto)
- `--no-verify` — Skip solution verification before submission (slightly faster)

### Reporting

- `--short-secs N` — Short report interval in seconds (default: 2)
- `--detailed-secs N` — Detailed report interval in seconds (default: 60)
- `--avg-window N` — Average calculation window in seconds (default: 300)

### Advanced

- `--log-csv FILE` — Log statistics to CSV file (framework ready)

## Environment Variables

- `NAME_PREFIX` — Same as `--name`
- `WORKERS` — Same as `--workers`
- `GPU_WORKERS` — Same as `--gpu-workers`
- `GPU_BATCH_SIZE` — Hashes per GPU kernel dispatch (default: 1048576)
- `REPORT_SHORT_INTERVAL` — Same as `--short-secs`
- `REPORT_DETAILED_INTERVAL` — Same as `--detailed-secs`
- `REPORT_AVERAGE_WINDOW` — Same as `--avg-window`
- `NOTIFY=1` — Enable desktop notifications (requires notify-send)

## Examples

### Basic Usage

```bash
# CPU mining with default settings
./target/release/rust_miner --name MyMiner

# GPU mining with 2 workers
./target/release/rust_miner --name MyMiner --gpu

# Specific number of CPU threads
./target/release/rust_miner --name MyMiner --workers 8
```

### Optimized Configurations

```bash
# Aggressive GPU mining (high-end cards)
GPU_BATCH_SIZE=4194304 ./target/release/rust_miner --name AS --gpu --gpu-workers 4

# Quiet mode with less frequent reporting
./target/release/rust_miner --name AS --gpu --short-secs 10 --detailed-secs 300

# Development/testing (skip verification)
./target/release/rust_miner --name Dev --no-verify
```

### Tuning Examples

```bash
# Try different work-group sizes for your GPU
./target/release/rust_miner --name AS --gpu --work-group-size 128
./target/release/rust_miner --name AS --gpu --work-group-size 256
./target/release/rust_miner --name AS --gpu --work-group-size 512

# Pin to specific CPU cores (Linux)
taskset -c 0-7 ./target/release/rust_miner --name AS --workers 8

# Set CPU to performance mode (requires root)
sudo cpupower frequency-set -g performance
./target/release/rust_miner --name AS
```

## Features

- **Optimized CPU Mining:** 2-4× faster than baseline with zero-allocation hot path
- **GPU Acceleration:** 30× faster than CPU using OpenCL
- **Smart Batching:** Reduced memory transfers and atomic operation overhead
- **Solution Verification:** Automatic validation before submission
- **Real-time Statistics:** Instantaneous and average hash rates in MH/s
- **Parent Change Detection:** Automatic restart when blockchain updates

## Help

```bash
./target/release/rust_miner --help
```
