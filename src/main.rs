use rayon::prelude::*;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};
use std::io::{self, Write};
use std::env;
use chrono::Local;
// OpenCL (GPU) support
use ocl::{Platform, Context, Device, Kernel, Program, Queue, Buffer, flags};
// use itoa; // Nicht nötig, itoa::Buffer reicht

// --- Configuration ---
const BASE_URL: &str = "http://hash.h10a.de/";
// Name suffix is mandatory and always appended to the provided prefix
const NAME_SUFFIX: &str = "-B6";
const SPAM_FILTER: usize = 30;

// Helper to return a short timestamp string for logs
fn ts() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

fn fmt_duration_hms(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}

// Send a desktop notification using `notify-send` when NOTIFY env var is set to 1/true.
fn notify(message: &str) {
    if let Ok(v) = env::var("NOTIFY") {
        let enabled = v.eq_ignore_ascii_case("1") || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes");
        if !enabled {
            return;
        }
    } else {
        // Default: notifications disabled unless NOTIFY is explicitly set
        return;
    }

    // Try to call notify-send; ignore any errors silently.
    let _ = std::process::Command::new("notify-send")
        .arg("rust_miner")
        .arg(message)
        .spawn();
}


/// Holds shared state for all threads
struct MinerState {
    http_client: Client,
    latest_parent: Mutex<String>,
    difficulty: AtomicUsize,
    solution_found: AtomicBool,
}

/// Configuration for additional features
#[allow(dead_code)]
struct MinerConfig {
    csv_log_file: Option<String>,
    local_work_size: Option<usize>,
    verify_before_submit: bool,
}

fn parse_reporting_config_and_name() -> (String, Duration, Duration, Duration, bool, Option<usize>, Option<usize>, MinerConfig) {
    // Defaults
    let mut short_secs = 1u64;
    let mut detailed_secs = 60u64;
    let mut avg_window_secs = 300u64;
    let mut name_prefix: Option<String> = None;

    // Env vars
    if let Ok(s) = env::var("REPORT_SHORT_INTERVAL") {
        if let Ok(v) = s.parse::<u64>() { short_secs = v; }
    }
    if let Ok(s) = env::var("REPORT_DETAILED_INTERVAL") {
        if let Ok(v) = s.parse::<u64>() { detailed_secs = v; }
    }
    if let Ok(s) = env::var("REPORT_AVERAGE_WINDOW") {
        if let Ok(v) = s.parse::<u64>() { avg_window_secs = v; }
    }

    let mut use_gpu = false;
    let mut gpu_workers_arg: Option<usize> = None;
    let mut workers_arg: Option<usize> = None;
    
    // New feature flags
    let mut csv_log_file: Option<String> = None;
    let mut local_work_size: Option<usize> = None;
    let mut verify_before_submit = true;  // Default enabled

    // CLI args override env vars. Supported args:
    // --short-secs N, --detailed-secs N, --avg-window N, --name PREFIX, --gpu
    // --gpu-workers N, --workers N (or -w N)
    // --log-csv FILE, --work-group-size N, --no-verify
    // --help or -h prints usage
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => {
                eprintln!("Usage: [--name PREFIX] [OPTIONS]");
                eprintln!();
                eprintln!("Required:");
                eprintln!("  --name PREFIX           Miner name (suffix '{}' will be appended)", NAME_SUFFIX);
                eprintln!();
                eprintln!("Mining Options:");
                eprintln!("  --gpu                   Use OpenCL GPU mining");
                eprintln!("  --workers N, -w N       Number of CPU worker threads (default: all CPUs)");
                eprintln!("  --gpu-workers N         Number of GPU worker threads (default: 2)");
                eprintln!();
                eprintln!("Performance Tuning:");
                eprintln!("  --work-group-size N     OpenCL local work group size (default: auto)");
                eprintln!();
                eprintln!("Reporting:");
                eprintln!("  --short-secs N          Short report interval in seconds (default: 2)");
                eprintln!("  --detailed-secs N       Detailed report interval in seconds (default: 60)");
                eprintln!("  --avg-window N          Average window in seconds (default: 300)");
                eprintln!();
                eprintln!("Logging & Analysis:");
                eprintln!("  --log-csv FILE          Log statistics to CSV file");
                eprintln!();
                eprintln!("Advanced:");
                eprintln!("  --no-verify             Skip verification before submitting solutions");
                eprintln!();
                eprintln!("Environment Variables:");
                eprintln!("  NAME_PREFIX, WORKERS, GPU_WORKERS, GPU_BATCH_SIZE");
                eprintln!("  REPORT_SHORT_INTERVAL, REPORT_DETAILED_INTERVAL, REPORT_AVERAGE_WINDOW");
                eprintln!("  NOTIFY=1                Enable desktop notifications (requires notify-send)");
                std::process::exit(0);
            }
            "--log-csv" => {
                match args.next() {
                    Some(val) => csv_log_file = Some(val),
                    None => {
                        eprintln!("Error: --log-csv requires a filename. Example: --log-csv stats.csv");
                        std::process::exit(1);
                    }
                }
            }
            "--work-group-size" | "--local-work-size" => {
                match args.next() {
                    Some(val) => {
                        if let Ok(n) = val.parse::<usize>() {
                            local_work_size = Some(n.max(1));
                        } else {
                            eprintln!("Invalid value for --work-group-size: {}", val);
                            std::process::exit(1);
                        }
                    }
                    None => {
                        eprintln!("Error: --work-group-size requires a value. Example: --work-group-size 256");
                        std::process::exit(1);
                    }
                }
            }
            "--no-verify" => {
                verify_before_submit = false;
            }
            "--gpu" => {
                // enable GPU/OpenCL backend
                use_gpu = true;
            }
            "--gpu-workers" => {
                match args.next() {
                    Some(val) => { if let Ok(n) = val.parse::<usize>() { gpu_workers_arg = Some(n.max(1)); } else { eprintln!("Invalid value for --gpu-workers: {}", val); std::process::exit(1); } }
                    None => { eprintln!("Error: --gpu-workers requires a value. Example: --gpu-workers 4"); std::process::exit(1); }
                }
            }
            "--workers" | "--worker" | "-w" => {
                match args.next() {
                    Some(val) => { if let Ok(n) = val.parse::<usize>() { workers_arg = Some(n.max(1)); } else { eprintln!("Invalid value for --workers: {}", val); std::process::exit(1); } }
                    None => { eprintln!("Error: --workers requires a value. Example: --workers 4"); std::process::exit(1); }
                }
            }
            "--name" => {
                match args.next() {
                    Some(val) => name_prefix = Some(val),
                    None => {
                        eprintln!("Error: --name requires a value. Example: --name AS");
                        std::process::exit(1);
                    }
                }
            }
            "--short-secs" => {
                if let Some(val) = args.next() { if let Ok(v) = val.parse::<u64>() { short_secs = v; } }
            }
            "--detailed-secs" => {
                if let Some(val) = args.next() { if let Ok(v) = val.parse::<u64>() { detailed_secs = v; } }
            }
            "--avg-window" => {
                if let Some(val) = args.next() { if let Ok(v) = val.parse::<u64>() { avg_window_secs = v; } }
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Usage: [--name PREFIX] [--short-secs N] [--detailed-secs N] [--avg-window N]");
                std::process::exit(1);
            }
        }
    }

    // Clamp sensible minimums
    if short_secs == 0 { short_secs = 1; }
    if detailed_secs < short_secs { detailed_secs = short_secs; }
    if avg_window_secs < short_secs { avg_window_secs = short_secs; }

    // Also check environment for WORKERS and GPU_WORKERS if CLI didn't supply
    if workers_arg.is_none() {
        if let Ok(s) = env::var("WORKERS") {
            if let Ok(v) = s.parse::<usize>() { workers_arg = Some(v.max(1)); }
        }
    }

    if gpu_workers_arg.is_none() {
        if let Ok(s) = env::var("GPU_WORKERS") {
            if let Ok(v) = s.parse::<usize>() { gpu_workers_arg = Some(v.max(1)); }
        }
    }

    (
        // Build name: env var overrides default, CLI overrides env var
        {
            // Check env var if CLI didn't supply
            let prefix = if let Some(p) = name_prefix {
                p
            } else if let Ok(p) = env::var("NAME_PREFIX") {
                p
            } else {
                eprintln!("Error: NAME prefix is required. Provide with --name PREFIX or NAME_PREFIX env var.");
                std::process::exit(1);
            };
            // Ensure the mandatory suffix is present exactly once
            if prefix.ends_with(NAME_SUFFIX) {
                prefix
            } else {
                format!("{}{}", prefix, NAME_SUFFIX)
            }
        },
        Duration::from_secs(short_secs),
        Duration::from_secs(detailed_secs),
        Duration::from_secs(avg_window_secs),
        use_gpu,
        gpu_workers_arg,
        workers_arg,
        MinerConfig {
            csv_log_file,
            local_work_size,
            verify_before_submit,
        },
    )
}

fn main() {
    let (name, short_interval, detailed_interval, average_window, use_gpu, gpu_workers_cli, workers_cli, config) = parse_reporting_config_and_name();
    
    println!("{} Starting Rust miner for: {}", ts(), name);
    // Respect explicit worker count from CLI/env if provided, otherwise use all CPUs
    let num_miners = workers_cli.unwrap_or_else(|| num_cpus::get());
    println!("{} Using {} miner threads.", ts(), num_miners);

    println!(
        "{} Reporting: short={}s detailed={}s avg_window={}s",
        ts(),
        short_interval.as_secs(),
        detailed_interval.as_secs(),
        average_window.as_secs()
    );

    // Arc (Atomic Reference Counter) allows shared ownership across threads
    let state = Arc::new(MinerState {
        http_client: Client::new(),
        latest_parent: Mutex::new(String::new()),
        difficulty: AtomicUsize::new(0),
        solution_found: AtomicBool::new(false),
    });

    // --- Main Loop ---
    let mut current_parent =
        fetch_parent_and_difficulty(state.clone(), "MainThread").expect("Failed initial fetch");

    loop {
        // find_seed will block until a solution is found OR the parent changes
            let result = if use_gpu {
            find_seed_gpu(
                state.clone(),
                &current_parent,
                num_miners,
                short_interval,
                detailed_interval,
                average_window,
                &name,
                gpu_workers_cli,
            )
        } else {
            find_seed(
                state.clone(),
                &current_parent,
                num_miners,
                short_interval,
                detailed_interval,
                average_window,
                &name,
            )
        };

        match result {
            // A miner thread found a valid seed
            Some(seed) => {
                    println!("{} Solution found! Submitting seed: {}", ts(), seed);
                    current_parent = submit_seed(state.clone(), &current_parent, &seed, &name, config.verify_before_submit);
                }
            // The monitor thread found a new parent (aborted work)
            None => {
                    // Get the new parent that the monitor thread may have found
                    let new_parent = state.latest_parent.lock().unwrap().clone();
                    if new_parent != current_parent {
                        println!("{} Parent changed. Restarting work.", ts());
                        current_parent = new_parent;
                    } else {
                        // Guard against tight-loop/spam if monitor signalled but parent wasn't updated.
                        // Re-fetch parent (this also updates shared state) and wait a bit before retrying.
                        println!("{} Monitor signalled restart but parent unchanged. Re-fetching and sleeping briefly to avoid spam...", ts());
                        current_parent = fetch_parent_and_difficulty(state.clone(), "MainThread").unwrap_or_else(|| current_parent.clone());
                        thread::sleep(Duration::from_secs(2));
                    }
            }
        }

        // Safety check if parent is empty
       if current_parent.is_empty() {
           println!("{} Parent is empty, re-fetching...", ts());
           current_parent = fetch_parent_and_difficulty(state.clone(), "MainThread").unwrap_or_default();
           thread::sleep(Duration::from_secs(2));
       }
    }
}

/**
 * Fetches the latest parent hash and difficulty from the server.
 * This maps to your `getParentAndDifficulty` method.
 */
fn fetch_parent_and_difficulty(state: Arc<MinerState>, thread_name: &str) -> Option<String> {
    if thread_name == "MainThread" {
        // Enforce server delay for the main loop
        thread::sleep(Duration::from_millis(2100));
        println!("\n{} Fetching parent and difficulty...", ts());
    }

    let url = format!("{}?raw", BASE_URL);
    match state.http_client.get(&url).send() {
        Ok(response) => {
            let text = response.text().ok()?;
            let mut lines = text.lines();

            // 1. Parse Difficulty
            if let Some(diff_line) = lines.next() {
                if let Ok(new_diff) = diff_line.parse::<usize>() {
                    let old_diff = state.difficulty.swap(new_diff, Ordering::SeqCst);
                    if new_diff != old_diff && old_diff != 0 {
                        println!("Difficulty updated: {}", new_diff);
                    }
                }
            }

            // 2. Parse Parent Hash
            let mut best_parent = String::new();
            let mut max_level = -1;

            for line in lines {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    if let Ok(level) = parts[1].parse::<i32>() {
                        if level >= max_level {
                            max_level = level;
                            best_parent = parts[0].to_string();
                        }
                    }
                }
            }

            if !best_parent.is_empty() {
                // Update shared state
                let mut parent_lock = state.latest_parent.lock().unwrap();
                *parent_lock = best_parent.clone();
                Some(best_parent)
            } else {
                None
            }
        }
        Err(e) => {
            if thread_name == "MainThread" {
                eprintln!("{} Failed to get parent: {}. Exiting.", ts(), e);
                std::process::exit(1);
            } else {
                eprintln!("{} Monitor failed to fetch: {}", ts(), e);
            }
            None
        }
    }
}

/**
 * Verify a solution before submitting
 */
fn verify_solution(parent: &str, name: &str, seed: &str, required_bits: usize) -> Result<usize, String> {
    let message = format!("{} {} {}", parent, name, seed);
    let mut hasher = Sha256::new();
    hasher.update(message.as_bytes());
    let hash = hasher.finalize();
    let bits = count_zerobits(&hash);
    
    if bits >= required_bits {
        Ok(bits)
    } else {
        Err(format!("Solution has {} bits but {} required", bits, required_bits))
    }
}

/**
 * Submits a winning seed to the server.
 * This maps to your (missing) `sendSeed` method.
 */
fn submit_seed(state: Arc<MinerState>, parent: &str, seed: &str, name: &str, verify: bool) -> String {
    // Verify solution before submitting if enabled
    if verify {
        let required = state.difficulty.load(Ordering::SeqCst);
        match verify_solution(parent, name, seed, required) {
            Ok(bits) => {
                println!("{} ✓ Solution verified: {} bits (required: {})", ts(), bits, required);
            }
            Err(e) => {
                eprintln!("{} ✗ VERIFICATION FAILED: {}", ts(), e);
                eprintln!("{} Parent: {}, Name: {}, Seed: {}", ts(), parent, name, seed);
                // Still try to fetch new parent
                return fetch_parent_and_difficulty(state, "MainThread").unwrap_or_default();
            }
        }
    }
    
    let url = format!(
        "{}?Z={}&P={}&R={}",
        BASE_URL, parent, name, seed
    );
    println!("{} Submitting to: {}", ts(), url);

    match state.http_client.get(&url).send() {
        Ok(response) => {
            if let Ok(new_parent) = response.text() {
                if !new_parent.is_empty() {
                    println!("{} Server accepted. New parent: {}", ts(), new_parent);
                    // Update shared state
                    *state.latest_parent.lock().unwrap() = new_parent.clone();
                    return new_parent;
                }
            }
            println!("{} Server gave an empty response.", ts());
        }
        Err(e) => {
            eprintln!("{} Failed to submit seed: {}", ts(), e);
        }
    }
    // If submission failed, re-fetch the parent manually
    println!("{} Submission failed. Re-fetching parent.", ts());
    fetch_parent_and_difficulty(state, "MainThread").unwrap_or_default()
}

/**
 * Spawns miner threads and a monitor thread to find the next seed.
 * Returns `Some(seed)` if a miner wins.
 * Returns `None` if the monitor thread finds a new parent first.
 */
fn find_seed(
    state: Arc<MinerState>,
    parent: &str,
    num_miners: usize,
    short_interval: Duration,
    detailed_interval: Duration,
    average_window: Duration,
    name: &str,
) -> Option<String> {
    println!(
        "{} Starting {} miners on parent: ...{}  {}",
        ts(),
        num_miners,
        &parent[parent.len() - 10..],
        Local::now().format("%H:%M:%S")
    );

    // Reset the "solution found" flag for this round
    state.solution_found.store(false, Ordering::SeqCst);

    // --- Progress reporting shared state ---
    // Total hashes attempted across all miners this round
    let total_hashes = Arc::new(AtomicUsize::new(0));
    // Best overall zero-bit count seen this round
    let best_overall = Arc::new(AtomicUsize::new(0));
    // Per-thread bests for occasional detailed reporting
    let best_per_thread = Arc::new(Mutex::new(vec![0usize; num_miners]));

    // `mpsc` is a (Multiple Producer, Single Consumer) channel.
    // Miners will be producers (tx), the main thread is the consumer (rx).
    let (tx, rx) = mpsc::channel();

    // capture round start time so reporter can show elapsed time
    let round_start = Instant::now();

    // We use `rayon::scope` to block until all threads in it are finished.
    let _result = rayon::scope(move |s| {
        // --- 1. Spawn the Monitor Thread ---
        let monitor_state = state.clone();
        let parent_clone = parent.to_string();
        s.spawn(move |_| {
            while !monitor_state.solution_found.load(Ordering::SeqCst) {
                // Check server every 3 seconds
                thread::sleep(Duration::from_secs(3));

                // If we're already stopping, don't bother fetching
                if monitor_state.solution_found.load(Ordering::SeqCst) {
                    break;
                }

                if let Some(new_parent) = fetch_parent_and_difficulty(monitor_state.clone(), "MonitorThread") {
                    if new_parent != parent_clone {
                        // Console-first notification: always print a prominent, timestamped banner
                        println!("{} ===================== BLOCK MINED BY SOMEONE ELSE =====================", ts());
                        println!("{} New parent: {}", ts(), new_parent);
                        println!("{} ======================================================================", ts());

                        // Optional desktop notification if requested
                        if !monitor_state.solution_found.load(Ordering::SeqCst) {
                            let msg = format!("Block mined by someone else. New parent: {}", &new_parent);
                            notify(&msg);
                        }

                        // Update shared parent so main loop can pick it up
                        if let Ok(mut lock) = monitor_state.latest_parent.lock() {
                            *lock = new_parent.clone();
                        }
                        // Signal all miners to stop
                        monitor_state.solution_found.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
        });

        // --- 1.5 Spawn the Reporter Thread ---
        let reporter_state = state.clone();
        let reporter_total = total_hashes.clone();
        let reporter_best_overall = best_overall.clone();
        let reporter_bests = best_per_thread.clone();
        s.spawn(move |_| {
            let mut last_total = 0usize;
            let mut last_instant = Instant::now();
            let mut window_start = Instant::now();
            let mut window_start_total = reporter_total.load(Ordering::SeqCst);
            let mut detailed_acc = Duration::ZERO;

            while !reporter_state.solution_found.load(Ordering::SeqCst) {
                thread::sleep(short_interval);
                let now = Instant::now();
                let total = reporter_total.load(Ordering::SeqCst);
                let delta = total.saturating_sub(last_total);
                let elapsed = now.duration_since(last_instant).as_secs_f64();
                // instantaneous delta rate (since last short interval)
                let inst_rate = if elapsed > 0.0 { (delta as f64) / elapsed } else { 0.0 };
                last_total = total;
                last_instant = now;

                // sliding window average rate
                let window_elapsed = now.duration_since(window_start).as_secs_f64();
                if window_elapsed > 0.0 {
                    // if window too old, reset window
                    if window_elapsed > average_window.as_secs_f64() {
                        window_start = now;
                        window_start_total = total;
                    }
                }
                let window_elapsed = now.duration_since(window_start).as_secs_f64();
                let window_total = total.saturating_sub(window_start_total);
                let avg_rate = if window_elapsed > 0.0 { (window_total as f64) / window_elapsed } else { inst_rate };

                // Short live update (concise) - convert to MH/s for readability
                let best = reporter_best_overall.load(Ordering::SeqCst);
                let required = reporter_state.difficulty.load(Ordering::SeqCst);
                let avg_mh = avg_rate / 1e6_f64;
                let inst_mh = inst_rate / 1e6_f64;
                // report elapsed time since round start in brackets
                let elapsed_since_round = fmt_duration_hms(now.duration_since(round_start));
                print!(
                    "{} [{}] total: {}  avg: {:.3} MH/s  inst: {:.3} MH/s  best: {} bits  req: {}    \r",
                    ts(), elapsed_since_round, total, avg_mh, inst_mh, best, required
                );
                io::stdout().flush().ok();

                // Accumulate to occasionally print a detailed snapshot
                detailed_acc += short_interval;
                if detailed_acc >= detailed_interval {
                    detailed_acc = Duration::ZERO;
                    println!(); // newline after the concise overwrites
                    let per = reporter_bests.lock().unwrap();
                    let avg_mh = avg_rate / 1e6_f64;
                    println!(
                        "{} Detailed report at {}: total hashes: {}  avg: {:.3} MH/s  best overall: {} bits  required: {}",
                        ts(),
                        Local::now().format("%Y-%m-%d %H:%M:%S"),
                        total,
                        avg_mh,
                        best,
                        required
                    );
                    for (i, b) in per.iter().enumerate() {
                        println!("  miner {:2}: best {} bits", i, b);
                    }
                    println!("--- end detailed report ---");
                }
            }
            // Ensure next prints don't keep overwriting
            println!();
        });

        // --- 2. Spawn the Miner Threads ---
        let name_spacer_bytes = format!(" {} ", name).into_bytes();
        let parent_bytes = parent.as_bytes();
        let initial_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        (0..num_miners).into_par_iter().for_each_with(tx, |tx, thread_id| {
            // Pre-compute the base message that's constant across all iterations
            let mut base_msg = Vec::with_capacity(parent_bytes.len() + name_spacer_bytes.len() + 32);
            base_msg.extend_from_slice(parent_bytes);
            base_msg.extend_from_slice(&name_spacer_bytes);
            let base_len = base_msg.len();

            let mut current_seed = initial_seed + thread_id as u64;
            let mut best_hash_count = 0;
            
            // Batch atomic updates to reduce cache line bouncing
            const BATCH_SIZE: usize = 1000;
            let mut local_hash_count = 0;
            let mut local_best_update_pending = false;

            // Reusable buffers to avoid allocations
            let mut seed_buffer = itoa::Buffer::new();
            let mut hasher = Sha256::new();

            // This is the main mining loop for one thread
            while !state.solution_found.load(Ordering::SeqCst) {
                current_seed += num_miners as u64;
                
                // Format seed directly into reusable buffer
                let seed_bytes = seed_buffer.format(current_seed).as_bytes();
                
                // Build complete message (reuse base_msg capacity)
                base_msg.truncate(base_len);
                base_msg.extend_from_slice(seed_bytes);
                
                // Hash in one go (faster than cloning)
                hasher.update(&base_msg);
                let hash = hasher.finalize_reset();  // finalize_reset reuses hasher state

                // Count zero bits
                let count = count_zerobits(&hash);
                let required_diff = state.difficulty.load(Ordering::SeqCst);
                
                local_hash_count += 1;

                if count >= required_diff {
                    // --- Found a solution! ---
                    // Flush pending stats before claiming solution
                    if local_hash_count > 0 {
                        total_hashes.fetch_add(local_hash_count, Ordering::Relaxed);
                    }
                    
                    // Try to "claim" the solution.
                    if state.solution_found.compare_exchange(
                        false, true, Ordering::SeqCst, Ordering::Relaxed
                    ).is_ok() {
                        // We were the first! Send the winning seed.
                        println!(
                            "{} Miner {} DONE: {} {}",
                            ts(),
                            thread_id,
                            count,
                            hex::encode(&hash)
                        );
                        let seed_string = current_seed.to_string();
                        tx.send(seed_string).unwrap();
                    }
                    // Break whether we were first or not
                    break;
                } else if count > best_hash_count {
                    best_hash_count = count;
                    local_best_update_pending = true;
                    
                    // Try updating best overall (less frequently to reduce contention)
                    let prev = best_overall.load(Ordering::Relaxed);
                    if count > prev {
                        // Try to update if still better (single attempt, no loop)
                        let _ = best_overall.compare_exchange_weak(
                            prev, count, Ordering::Relaxed, Ordering::Relaxed
                        );
                    }

                    if count >= SPAM_FILTER {
                        println!(
                            "{} Miner {} Best: {}/{} {}",
                            ts(),
                            thread_id,
                            count,
                            required_diff,
                            hex::encode(&hash)
                        );
                    }
                }
                
                // Batch update global counters to reduce atomic overhead
                if local_hash_count >= BATCH_SIZE {
                    total_hashes.fetch_add(local_hash_count, Ordering::Relaxed);
                    local_hash_count = 0;
                    
                    // Update per-thread best less frequently
                    if local_best_update_pending {
                        if let Ok(mut per) = best_per_thread.try_lock() {
                            per[thread_id] = best_hash_count;
                            local_best_update_pending = false;
                        }
                    }
                }
            }
            
            // Flush remaining counts
            if local_hash_count > 0 {
                total_hashes.fetch_add(local_hash_count, Ordering::Relaxed);
            }
            if local_best_update_pending {
                if let Ok(mut per) = best_per_thread.lock() {
                    per[thread_id] = best_hash_count;
                }
            }
        });
    });

    // `rayon::scope` has finished, so all miners and the monitor have stopped.
    // We now check if a seed was successfully sent to the channel.
    // `try_recv()` is non-blocking.
    match rx.try_recv() {
        Ok(winning_seed) => Some(winning_seed), // A miner won
        Err(_) => None,                     // No seed sent, so the monitor must have aborted
    }
}

fn count_zerobits(hash: &[u8]) -> usize {
    let mut bits = 0;
    for &byte in hash.iter() {
        if byte == 0 {
            bits += 8;
        } else {
            bits += byte.leading_zeros() as usize;
            return bits;
        }
    }
    bits
}

// --- GPU (OpenCL) POC implementation ---
fn find_seed_gpu(
    state: Arc<MinerState>,
    parent: &str,
    _num_miners: usize,
    short_interval: Duration,
    detailed_interval: Duration,
    average_window: Duration,
    name: &str,
    gpu_workers_cli: Option<usize>,
) -> Option<String> {
    println!("Starting GPU (OpenCL) miner for parent ...{}", &parent[parent.len()-10..]);
    // Reset the "solution found" flag for this GPU round
    state.solution_found.store(false, Ordering::SeqCst);

    // Try to find an OpenCL device
    let devices = match Device::list_all(Platform::default()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("OpenCL device query failed: {}. Falling back to CPU.", e);
            return find_seed(state, parent, num_cpus::get(), short_interval, detailed_interval, average_window, name);
        }
    };
    if devices.is_empty() {
        eprintln!("No OpenCL devices found. Falling back to CPU.");
        return find_seed(state, parent, num_cpus::get(), short_interval, detailed_interval, average_window, name);
    }

    // Choose first device
    let device = devices[0];
    println!("Using OpenCL device: {}", device.name().unwrap_or_else(|_| "unknown".into()));

    // Create context and queue
    let context = match Context::builder().devices(device).build() {
        Ok(c) => c,
        Err(e) => { eprintln!("Failed to create OpenCL context: {}. Falling back to CPU.", e); return find_seed(state, parent, num_cpus::get(), short_interval, detailed_interval, average_window, name); }
    };
    let queue = match Queue::new(&context, device, None) {
        Ok(q) => q,
        Err(e) => { eprintln!("Failed to create OpenCL queue: {}. Falling back to CPU.", e); return find_seed(state, parent, num_cpus::get(), short_interval, detailed_interval, average_window, name); }
    };

    // Optimized OpenCL SHA-256 kernel with GPU-side message construction and result filtering
    let src = r#"
/*
 * OpenCL SHA-256 Kernel
 *
 * This code runs on the GPU. It includes a full SHA-256 implementation
 * and a helper to convert a ulong (u64) nonce into a hex string,
 * which is required by the game's hash format.
 */

// --- SHA-256 Implementation (for OpenCL) ---
// (This is a standard, compact SHA-256 implementation adapted for OpenCL)

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

typedef struct {
    uchar M[64];
    uint H[8];
    ulong N;
} sha256_ctx;

constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// --- End of SHA-256 ---


// Helper to convert a ulong nonce to a 1-16 char hex string
inline int ulong_to_hex(ulong n, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    char buf[16];
    int i = 0;

    if (n == 0) {
        out[0] = '0';
        return 1;
    }

    while (n > 0) {
        buf[i++] = hex_chars[n & 0xF];
        n >>= 4;
    }

    for (int j = 0; j < i; j++) {
        out[j] = buf[i - 1 - j];
    }
    return i;
}


// SHA-256 transform function
void sha256_transform(sha256_ctx *ctx) {
    uint W[64];
    uint a, b, c, d, e, f, g, h;
    
    #pragma unroll
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        W[i] = (ctx->M[j] << 24) | (ctx->M[j + 1] << 16) | (ctx->M[j + 2] << 8) | ctx->M[j + 3];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    a = ctx->H[0]; b = ctx->H[1]; c = ctx->H[2]; d = ctx->H[3];
    e = ctx->H[4]; f = ctx->H[5]; g = ctx->H[6]; h = ctx->H[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint T1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        uint T2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    ctx->H[0] += a; ctx->H[1] += b; ctx->H[2] += c; ctx->H[3] += d;
    ctx->H[4] += e; ctx->H[5] += f; ctx->H[6] += g; ctx->H[7] += h;
}

// Main SHA-256 function
void sha256(const uchar* data, int len, uchar* hash_out) {
    sha256_ctx ctx;
    
    ctx.H[0] = 0x6a09e667; ctx.H[1] = 0xbb67ae85; ctx.H[2] = 0x3c6ef372; ctx.H[3] = 0xa54ff53a;
    ctx.H[4] = 0x510e527f; ctx.H[5] = 0x9b05688c; ctx.H[6] = 0x1f83d9ab; ctx.H[7] = 0x5be0cd19;
    ctx.N = 0;

    int pos = 0;
    while(pos < len) {
        int copy_len = min(64 - (int)(ctx.N % 64), len - pos);
        
        for(int i=0; i < copy_len; i++) {
            ctx.M[(ctx.N % 64) + i] = data[pos + i];
        }

        ctx.N += copy_len;
        pos += copy_len;

        if ((ctx.N % 64) == 0) {
            sha256_transform(&ctx);
        }
    }

    // Padding
    int n_mod_64 = (int)(ctx.N % 64);
    ctx.M[n_mod_64++] = 0x80;
    
    if (n_mod_64 > 56) {
        for(int i=n_mod_64; i < 64; i++) {
            ctx.M[i] = 0;
        }
        sha256_transform(&ctx);
        n_mod_64 = 0;
    }
    
    for(int i=n_mod_64; i < 56; i++) {
        ctx.M[i] = 0;
    }
    
    ulong n_bits = ctx.N * 8;
    ctx.M[56] = (uchar)(n_bits >> 56);
    ctx.M[57] = (uchar)(n_bits >> 48);
    ctx.M[58] = (uchar)(n_bits >> 40);
    ctx.M[59] = (uchar)(n_bits >> 32);
    ctx.M[60] = (uchar)(n_bits >> 24);
    ctx.M[61] = (uchar)(n_bits >> 16);
    ctx.M[62] = (uchar)(n_bits >> 8);
    ctx.M[63] = (uchar)(n_bits);

    sha256_transform(&ctx);
    
    for(int i = 0; i < 8; i++) {
        hash_out[i*4 + 0] = (uchar)(ctx.H[i] >> 24);
        hash_out[i*4 + 1] = (uchar)(ctx.H[i] >> 16);
        hash_out[i*4 + 2] = (uchar)(ctx.H[i] >> 8);
        hash_out[i*4 + 3] = (uchar)(ctx.H[i]);
    }
}


// Helper to check for N leading zero bits
inline int count_zero_bits_kernel(const uchar* hash, int difficulty) {
    int bits = 0;
    for (int i = 0; i < 32; i++) {
        uchar byte = hash[i];
        if (byte == 0) {
            bits += 8;
        } else {
            uchar b = byte;
            while ((b & 0x80) == 0) {
                bits++;
                b <<= 1;
            }
            break;
        }
        if (bits >= difficulty) break;
    }
    return bits;
}


/*
 * == The Main Kernel ==
 */
__kernel void find_hash_kernel(
    __global const uchar* base_line,
    int base_len,
    ulong start_nonce,
    uint difficulty,
    __global uint* result_local_id,
    __global uchar* result_hash,
    __global uint* best_bits_global
) {
    ulong global_id = get_global_id(0);
    ulong nonce = start_nonce + global_id;

    // --- Optimization: Check if already found before doing any work ---
    if (*result_local_id != (uint)(-1)) {
        return;
    }

    // 1. Construct the input string: "parent_hash name nonce_hex"
    uchar line[256];
    
    for(int i=0; i < base_len; i++) {
        line[i] = base_line[i];
    }

    char nonce_hex[17];
    int nonce_len = ulong_to_hex(nonce, nonce_hex);

    for(int i=0; i < nonce_len; i++) {
        line[base_len + i] = nonce_hex[i];
    }
    
    int total_len = base_len + nonce_len;

    // 2. Hash the constructed line
    uchar hash_output[32];
    sha256(line, total_len, hash_output);

    // 3. Check if the hash meets the difficulty
    int bits = count_zero_bits_kernel(hash_output, difficulty);

    // Update global best atomically
    atomic_max(best_bits_global, (uint)bits);

    if (bits >= difficulty) {
        // We found a winner!
        // Atomically write our global_id (if no one else has)
        if (atomic_cmpxchg(result_local_id, (uint)(-1), (uint)global_id) == (uint)(-1)) {
            // We were first! Write our hash to the output buffer.
            for(int i=0; i < 32; i++) {
                result_hash[i] = hash_output[i];
            }
        }
    }
}
"#;

    // Compile program
    let program = match Program::builder().src(src).build(&context) {
        Ok(p) => p,
        Err(e) => { eprintln!("Failed to build OpenCL program: {}. Falling back to CPU.", e); return find_seed(state, parent, num_cpus::get(), short_interval, detailed_interval, average_window, name); }
    };
    println!("OpenCL kernel compiled successfully");

    // Batch settings (tunable via env): number of seeds per kernel dispatch and number of parallel GPU worker threads
    let env_batch = env::var("GPU_BATCH_SIZE").ok().and_then(|s| s.parse::<usize>().ok());
    let env_workers = env::var("GPU_WORKERS").ok().and_then(|s| s.parse::<usize>().ok());
    let batch_size: usize = env_batch.unwrap_or(1048576);  // Increased from 16K to 1M for better GPU utilization
    // Preference order: CLI (--gpu-workers) -> env GPU_WORKERS -> default 2
    let gpu_workers: usize = gpu_workers_cli.or(env_workers).unwrap_or(2).max(1);

    // Prepare shared reporting/best arrays sized to gpu_workers
    let total_hashes = Arc::new(AtomicUsize::new(0));
    let best_overall = Arc::new(AtomicUsize::new(0));
    let best_per_thread = Arc::new(Mutex::new(vec![0usize; gpu_workers]));

    // Capture round start time for elapsed time reporting
    let round_start = Instant::now();

    // Simple monitor thread to detect parent updates
    let monitor_state = state.clone();
    let parent_clone = parent.to_string();
    thread::spawn(move || {
        while !monitor_state.solution_found.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_secs(3));
            if monitor_state.solution_found.load(Ordering::SeqCst) { break; }
            if let Some(new_parent) = fetch_parent_and_difficulty(monitor_state.clone(), "MonitorThread") {
                if new_parent != parent_clone {
                    // Store the found parent into shared state for main loop
                    if let Ok(mut lock) = monitor_state.latest_parent.lock() {
                        *lock = new_parent.clone();
                    }
                    monitor_state.solution_found.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
    });

    // Reporter thread (mirrors CPU reporter behavior)
    let reporter_state = state.clone();
    let reporter_total = total_hashes.clone();
    let reporter_best_overall = best_overall.clone();
    let reporter_bests = best_per_thread.clone();
    thread::spawn(move || {
        let mut last_total = 0usize;
        let mut last_instant = Instant::now();
        let mut window_start = Instant::now();
        let mut window_start_total = reporter_total.load(Ordering::SeqCst);
        let mut detailed_acc = Duration::ZERO;

        while !reporter_state.solution_found.load(Ordering::SeqCst) {
            thread::sleep(short_interval);
            let now = Instant::now();
            let total = reporter_total.load(Ordering::SeqCst);
            let delta = total.saturating_sub(last_total);
            let elapsed = now.duration_since(last_instant).as_secs_f64();
            let inst_rate = if elapsed > 0.0 { (delta as f64) / elapsed } else { 0.0 };
            last_total = total;
            last_instant = now;

            let window_elapsed = now.duration_since(window_start).as_secs_f64();
            if window_elapsed > 0.0 {
                if window_elapsed > average_window.as_secs_f64() {
                    window_start = now;
                    window_start_total = total;
                }
            }
            let window_elapsed = now.duration_since(window_start).as_secs_f64();
            let window_total = total.saturating_sub(window_start_total);
            let avg_rate = if window_elapsed > 0.0 { (window_total as f64) / window_elapsed } else { inst_rate };

            let best = reporter_best_overall.load(Ordering::SeqCst);
            let required = reporter_state.difficulty.load(Ordering::SeqCst);
            let avg_mh = avg_rate / 1e6_f64;
            let inst_mh = inst_rate / 1e6_f64;
            let elapsed_since_round = fmt_duration_hms(now.duration_since(round_start));
            print!(
                "{} [{}] total: {}  avg: {:.3} MH/s  inst: {:.3} MH/s  best: {} bits  req: {}    \r",
                ts(), elapsed_since_round, total, avg_mh, inst_mh, best, required
            );
            io::stdout().flush().ok();

            detailed_acc += short_interval;
            if detailed_acc >= detailed_interval {
                detailed_acc = Duration::ZERO;
                println!();
                let per = reporter_bests.lock().unwrap();
                let avg_mh = avg_rate / 1e6_f64;
                    println!(
                        "{} Detailed report at {}: total hashes: {}  avg: {:.3} MH/s  best overall: {} bits  required: {}",
                        ts(),
                        Local::now().format("%Y-%m-%d %H:%M:%S"),
                        total,
                        avg_mh,
                        best,
                        required
                    );
                for (i, b) in per.iter().enumerate() {
                    println!("  gpu_worker {:2}: best {} bits", i, b);
                }
                println!("--- end detailed report ---");
            }
        }
        println!();
    });
    // reporter started

    // Per-worker launch: each worker has its own buffers and kernel to avoid host-side synchronization
    let mut handles = Vec::with_capacity(gpu_workers);
    // Construct base_line: "parent_hash name " (with trailing space before nonce)
    let base_line = format!("{} {} ", parent, name);
    let base_line_bytes = base_line.as_bytes().to_vec();
    let program = program.clone();

    for worker_id in 0..gpu_workers {
        let queue = queue.clone();
        let base_line_bytes = base_line_bytes.clone();
        let state = state.clone();
        let total_hashes = total_hashes.clone();
        let best_overall = best_overall.clone();
        let best_per_thread = best_per_thread.clone();
        let program = program.clone();

        let handle = thread::spawn(move || {
            // Prepare persistent buffers per worker
            let base_line_buf: Buffer<u8> = Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
                .len(base_line_bytes.len())
                .copy_host_slice(&base_line_bytes)
                .build().unwrap();

            // Result buffers for the new kernel
            // result_local_id: single u32, initialized to -1 (0xFFFFFFFF)
            let result_local_id_buf: Buffer<u32> = Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(1)
                .build().unwrap();

            // result_hash: 32 bytes
            let result_hash_buf: Buffer<u8> = Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_WRITE_ONLY)
                .len(32)
                .build().unwrap();

            // best_bits_global: single u32 to track best across all work items
            let best_bits_buf: Buffer<u32> = Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(1)
                .build().unwrap();

            let kernel = Kernel::builder()
                .program(&program)
                .name("find_hash_kernel")
                .queue(queue.clone())
                .global_work_size(batch_size)
                .arg(&base_line_buf)
                .arg(base_line_bytes.len() as i32)
                .arg(0u64)  // start_nonce placeholder
                .arg(0u32)  // difficulty placeholder
                .arg(&result_local_id_buf)
                .arg(&result_hash_buf)
                .arg(&best_bits_buf)
                .build().unwrap();

            let mut result_local_id_host = vec![0xFFFFFFFFu32; 1];
            let mut result_hash_host = vec![0u8; 32];
            let mut best_bits_host = vec![0u32; 1];

            // Nonce base per worker to avoid overlap
            let mut nonce_base: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64 + (worker_id as u64 * 1000000);

            while !state.solution_found.load(Ordering::SeqCst) {
                let required = state.difficulty.load(Ordering::SeqCst);
                
                // Reset result_local_id to -1 (no solution found yet)
                result_local_id_host[0] = 0xFFFFFFFF;
                result_local_id_buf.write(&result_local_id_host).enq().unwrap();

                // Reset best_bits to 0 for this batch
                best_bits_host[0] = 0;
                best_bits_buf.write(&best_bits_host).enq().unwrap();

                // Set kernel arguments for this batch
                kernel.set_arg(2, nonce_base).unwrap();
                kernel.set_arg(3, required as u32).unwrap();

                // Launch kernel
                unsafe { kernel.enq().unwrap(); }

                // Read back result_local_id and best_bits
                result_local_id_buf.read(&mut result_local_id_host).enq().unwrap();
                best_bits_buf.read(&mut best_bits_host).enq().unwrap();
                queue.finish().unwrap();  // Wait for kernel + reads to complete

                // Update stats
                total_hashes.fetch_add(batch_size, Ordering::Relaxed);
                
                // Update best from GPU
                let gpu_best = best_bits_host[0] as usize;
                if gpu_best > best_overall.load(Ordering::SeqCst) {
                    best_overall.store(gpu_best, Ordering::SeqCst);
                    if let Ok(mut per) = best_per_thread.lock() {
                        per[worker_id] = gpu_best;
                    }
                }
                
                // Check if a solution was found
                if result_local_id_host[0] != 0xFFFFFFFF {
                    // Solution found! Read the hash
                    result_hash_buf.read(&mut result_hash_host).enq().unwrap();
                    queue.finish().unwrap();
                    
                    // Calculate the winning nonce
                    let winning_nonce = nonce_base + (result_local_id_host[0] as u64);
                    
                    // Count the bits in the hash for reporting
                    let bits = count_zerobits(&result_hash_host);
                    
                    if state.solution_found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                        // Format nonce as hex string (matching the game's format)
                        let nonce_string = format!("{:x}", winning_nonce);
                        println!("GPU worker {} found nonce {} with {} bits", worker_id, nonce_string, bits);
                        return Some(nonce_string);
                    }
                    break;
                }

                // Advance nonce base to avoid overlap with other workers
                nonce_base = nonce_base.wrapping_add((gpu_workers * batch_size) as u64);
                
                if state.solution_found.load(Ordering::SeqCst) { break; }
            }

            None::<String>
        });
        handles.push(handle);
    }

    // Wait for workers to finish and check for a result
    for h in handles {
        if let Ok(ret) = h.join() {
            if let Some(seed) = ret {
                return Some(seed);
            }
        }
    }

    None
}
