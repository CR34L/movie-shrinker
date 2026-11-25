//! Live system gauges for the footer, from stock macOS tools (no root):
//! `top` for CPU, `ioreg` for per-disk read/write bytes and GPU utilization.
//! The `top` child exits with SIGPIPE once the app quits and its stdout closes.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Stats {
    /// 100 − idle, across all cores.
    pub cpu_busy: f32,
    /// GPU "Device Utilization %" (graphics cores; the video encoder block
    /// isn't exposed without root).
    pub gpu_busy: f32,
    /// (read MB/s, write MB/s) per physical disk, keyed by BSD name, e.g. "disk10".
    pub disk_mbs: HashMap<String, (f32, f32)>,
}

pub fn start() -> Arc<Mutex<Stats>> {
    let stats = Arc::new(Mutex::new(Stats::default()));

    let s = stats.clone();
    std::thread::spawn(move || {
        let Ok(mut child) = Command::new("/usr/bin/top")
            .args(["-l", "0", "-s", "1", "-n", "0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        else {
            return;
        };
        for line in BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
            if let Some(idle) = parse_cpu_idle(&line) {
                s.lock().unwrap().cpu_busy = (100.0 - idle).clamp(0.0, 100.0);
            }
        }
        let _ = child.wait();
    });

    // Disk counters and GPU: poll ioreg once a second and diff the byte totals.
    let s = stats.clone();
    std::thread::spawn(move || {
        let mut prev: Option<(Instant, HashMap<String, (u64, u64)>)> = None;
        loop {
            let now = Instant::now();
            if let Some(cur) = ioreg(&["-r", "-c", "IOBlockStorageDriver", "-l", "-d", "2", "-w0"])
                .map(|t| parse_disk_bytes(&t))
            {
                if let Some((t0, old)) = &prev {
                    let secs = now.duration_since(*t0).as_secs_f32().max(0.1);
                    let mb = |a: u64, b: u64| a.saturating_sub(b) as f32 / 1_000_000.0 / secs;
                    let rates = cur
                        .iter()
                        .filter_map(|(d, &(r, w))| {
                            let &(r0, w0) = old.get(d)?;
                            Some((d.clone(), (mb(r, r0), mb(w, w0))))
                        })
                        .collect();
                    s.lock().unwrap().disk_mbs = rates;
                }
                prev = Some((now, cur));
            }
            if let Some(v) = ioreg(&["-r", "-d", "1", "-c", "IOAccelerator"]).and_then(|t| parse_gpu(&t)) {
                s.lock().unwrap().gpu_busy = v;
            }
            std::thread::sleep(Duration::from_secs(1).saturating_sub(now.elapsed()));
        }
    });

    stats
}

fn ioreg(args: &[&str]) -> Option<String> {
    let out = Command::new("/usr/sbin/ioreg").args(args).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// "CPU usage: 32.82% user, 22.78% sys, 44.39% idle"
fn parse_cpu_idle(line: &str) -> Option<f32> {
    let rest = line.strip_prefix("CPU usage:")?;
    rest.split(',').find(|p| p.contains("idle"))?.trim().split('%').next()?.trim().parse().ok()
}

/// Cumulative (read, write) bytes per disk. In `ioreg -l -d 2` output each
/// IOBlockStorageDriver's "Statistics" is followed by its media's "BSD Name".
fn parse_disk_bytes(text: &str) -> HashMap<String, (u64, u64)> {
    let num = |line: &str, key: &str| -> Option<u64> {
        let rest = line.split(&format!("\"{key}\"=")).nth(1)?;
        rest.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    let mut map = HashMap::new();
    let mut pending: Option<(u64, u64)> = None;
    for line in text.lines() {
        if line.contains("<class IOBlockStorageDriver") {
            pending = None;
        } else if line.contains("\"Statistics\" = {") {
            pending = num(line, "Bytes (Read)").zip(num(line, "Bytes (Write)"));
        } else if let Some(rest) = line.split("\"BSD Name\" = \"").nth(1) {
            if let (Some(bytes), Some(name)) = (pending.take(), rest.split('"').next()) {
                map.insert(name.to_string(), bytes);
            }
        }
    }
    map
}

/// Highest "Device Utilization %"=N among accelerators.
fn parse_gpu(text: &str) -> Option<f32> {
    text.split("\"Device Utilization %\"=")
        .skip(1)
        .filter_map(|r| r.split(|c: char| !c.is_ascii_digit()).next()?.parse::<f32>().ok())
        .reduce(f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_output() {
        assert_eq!(parse_cpu_idle("CPU usage: 32.82% user, 22.78% sys, 44.39% idle"), Some(44.39));

        let io = r#"+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x1>
    |   "Statistics" = {"Operations (Write)"=5,"Bytes (Read)"=1000,"Bytes (Write)"=2000,"Errors (Read)"=0}
+-o IOBlockStorageDriver  <class IOBlockStorageDriver, id 0x2>
    |   "Statistics" = {"Bytes (Read)"=93191886848,"Bytes (Write)"=483419291648}
    +-o Seagate Expansion SW Media  <class IOMedia, id 0x3>
          "BSD Name" = "disk10"
"#;
        let m = parse_disk_bytes(io);
        assert_eq!(m.len(), 1, "driver without media must not steal the next name");
        assert_eq!(m["disk10"], (93191886848, 483419291648));

        let gpu = r#"| "PerformanceStatistics" = {"Tiler Utilization %"=3,"Device Utilization %"=17,"x"=1}"#;
        assert_eq!(parse_gpu(gpu), Some(17.0));
    }
}
