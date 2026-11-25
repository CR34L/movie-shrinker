//! Shrinks one video file in place using ffmpeg (libx265 / libx264).
//!
//! The encode goes to a hidden temp file next to the source. Only if it
//! succeeds AND is smaller does it atomically replace the original (same
//! path, same name). Otherwise the original is left untouched.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::SystemTime;

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum MaxRes {
    P720,
    P1440,
}

impl MaxRes {
    /// (long side, short side)
    fn bounds(self) -> (u32, u32) {
        match self {
            MaxRes::P720 => (1280, 720),
            MaxRes::P1440 => (2560, 1440),
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            MaxRes::P720 => "720p",
            MaxRes::P1440 => "2K (1440p)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Codec {
    H265,
    H264,
}

impl Codec {
    pub fn label(self) -> &'static str {
        match self {
            Codec::H265 => "H.265 (smaller)",
            Codec::H264 => "H.264 (compatible)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Quality {
    High,
    Balanced,
    Small,
}

impl Quality {
    pub fn label(self) -> &'static str {
        match self {
            Quality::High => "High",
            Quality::Balanced => "Balanced",
            Quality::Small => "Smallest",
        }
    }
    fn crf(self, codec: Codec) -> u8 {
        let base = match codec {
            Codec::H265 => 24,
            Codec::H264 => 21,
        };
        match self {
            Quality::High => base - 3,
            Quality::Balanced => base,
            Quality::Small => base + 4,
        }
    }
}

/// Target bitrate (kbit/s) for VideoToolbox. Its constant-quality mode
/// spends far more bits than x265 (1.24 GB -> 828 MB on a real 1080p
/// recording), so hardware encodes get an explicit, resolution-sized budget.
fn vt_kbps(res: MaxRes, codec: Codec, quality: Quality) -> u32 {
    let base = match res {
        MaxRes::P720 => 1600.0,
        MaxRes::P1440 => 4500.0,
    };
    let q = match quality {
        Quality::High => 1.5,
        Quality::Balanced => 1.0,
        Quality::Small => 0.65,
    };
    let c = if codec == Codec::H264 { 1.5 } else { 1.0 };
    (base * q * c) as u32
}

/// Which hardware a single encode runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    /// x265 / x264 on the CPU cores.
    Cpu,
    /// Apple VideoToolbox media engine (GPU block).
    Gpu,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Cpu => "CPU",
            Engine::Gpu => "GPU",
        }
    }
}

/// Which engines the queue may use.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Mode {
    /// GPU and CPU lanes in parallel — fastest.
    All,
    /// Only x265/x264 — smallest files, consistent quality.
    CpuOnly,
    /// Only VideoToolbox — fast, leaves the CPU free.
    GpuOnly,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::All => "CPU + GPU",
            Mode::CpuOnly => "CPU only",
            Mode::GpuOnly => "GPU only",
        }
    }
    /// Number of parallel (cpu, gpu) encode lanes for this machine.
    pub fn lanes(self) -> (usize, usize) {
        let (cpu, gpu) = hardware_lanes();
        match self {
            Mode::All => (cpu, gpu),
            Mode::CpuOnly => (cpu, 0),
            Mode::GpuOnly => (0, gpu),
        }
    }
}

fn sysctl(name: &str) -> Option<String> {
    let out = Command::new("/usr/sbin/sysctl").args(["-n", name]).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

/// x265 scales to only ~3–5 threads per 720p/1440p stream, so several
/// streams side by side keep every core busy. Max/Ultra chips have 2+
/// hardware encode engines; others have one.
fn hardware_lanes() -> (usize, usize) {
    static L: OnceLock<(usize, usize)> = OnceLock::new();
    *L.get_or_init(|| {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        // ~1 lane per 3 cores; each lane gets a matching thread budget below
        let cpu = std::env::var("SHRINK_CPU_LANES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or((cores / 3).clamp(1, 8));
        let chip = sysctl("machdep.cpu.brand_string").unwrap_or_default();
        let gpu = if chip.contains("Ultra") {
            4
        } else if chip.contains("Max") {
            2
        } else {
            1
        };
        (cpu, gpu)
    })
}

/// Threads one CPU lane may use. Without this every x265 sizes its pool for
/// the whole machine, and 3 parallel encodes run ~160 threads on 10 cores:
/// the kernel spends ~25% of CPU just switching and each file crawls at ~1×.
/// Reserve ~1 core per GPU lane (it decodes/scales on the CPU) and split the rest.
fn cpu_lane_threads(mode: Mode) -> usize {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let (cpu, gpu) = mode.lanes();
    let budget = cores.saturating_sub(gpu).max(1);
    budget.div_ceil(cpu.max(1)).max(2)
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    pub max_res: MaxRes,
    pub codec: Codec,
    pub quality: Quality,
    pub mode: Mode,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_res: MaxRes::P720,
            codec: Codec::H265,
            quality: Quality::Balanced,
            mode: Mode::All,
        }
    }
}

pub enum Outcome {
    Replaced { old: u64, new: u64 },
    /// Encode finished but wasn't smaller; original kept.
    KeptOriginal { old: u64, new: u64 },
    /// Marked as shrunk earlier (or already small HEVC); not touched.
    AlreadyShrunk,
    Cancelled,
}

pub const EXTENSIONS: &[&str] = &["mp4", "m4v", "mkv", "mov"];

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Finder-launched apps don't get the shell PATH, so look in the usual places.
/// A copy bundled next to the executable (inside the .app) wins.
fn find_tool(name: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(name));
        }
    }
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"] {
        candidates.push(Path::new(dir).join(name));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join(name)));
    }
    candidates.into_iter().find(|p| p.is_file())
}

pub fn ffmpeg() -> Result<&'static Path, String> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| find_tool("ffmpeg"))
        .as_deref()
        .ok_or_else(|| "ffmpeg not found — install with: brew install ffmpeg".into())
}

/// Parses "  Duration: 00:17:12.40, start: ..." from ffmpeg's header log.
fn parse_duration_us(line: &str) -> Option<u64> {
    let t = line.trim_start().strip_prefix("Duration: ")?.split(',').next()?;
    let mut parts = t.split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let sec: f64 = parts.next()?.parse().ok()?;
    Some(((h * 3600.0 + m * 60.0 + sec) * 1_000_000.0) as u64)
}

/// Fit inside the bounding box, orientation-aware (portrait gets 720 wide),
/// never upscale, keep even dimensions. Evaluated on post-rotation frames.
fn scale_filter(res: MaxRes) -> String {
    let (l, s) = res.bounds();
    format!(
        "scale=w='if(gte(iw,ih),min(iw,{l}),min(iw,{s}))':h='if(gte(iw,ih),min(ih,{s}),min(ih,{l}))'\
         :force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap().to_string_lossy();
    let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();
    path.with_file_name(format!(".{name}.shrinking.{ext}"))
}

/// Removes the temp file on every exit path unless disarmed.
struct TempGuard(PathBuf);
impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn shrink(
    path: &Path,
    settings: Settings,
    engine: Engine,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(f32, f32),
) -> Result<Outcome, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let old_size = meta.len();

    // Stream the file into ffmpeg in big sequential chunks when the format
    // allows it: on a hard disk this replaces thousands of small seeking
    // reads (3 MB/s measured) with one straight read per file.
    let layout = layout(path);
    if already_shrunk(path, layout, settings.max_res) {
        return Ok(Outcome::AlreadyShrunk);
    }
    let stream = layout.streamable();
    let fed = std::sync::atomic::AtomicU64::new(0);

    let tmp = temp_path(path);
    let _guard = TempGuard(tmp.clone());
    let ext = path.extension().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
    let is_mp4 = ext != "mkv";

    let threads = match engine {
        Engine::Cpu => cpu_lane_threads(settings.mode),
        Engine::Gpu => 2,
    };
    let threads_s = threads.to_string();

    // CPU lanes run at lower priority so the GPU lanes (which need a little
    // CPU to feed the hardware encoder) and the desktop stay responsive.
    let mut cmd = match engine {
        Engine::Cpu => {
            let mut c = Command::new("/usr/bin/nice");
            c.args(["-n", "10"]).arg(ffmpeg()?);
            c
        }
        Engine::Gpu => Command::new(ffmpeg()?),
    };
    cmd.args(["-hide_banner", "-nostdin", "-y", "-loglevel", "info", "-nostats"])
        .args(["-progress", "pipe:1"])
        // decode on the media engine so CPU cores are free for encoding;
        // ffmpeg falls back to software decode (capped by -threads) if unsupported
        .args(["-hwaccel", "videotoolbox", "-threads", &threads_s])
        .arg("-i")
        .arg(if stream { Path::new("pipe:0") } else { path })
        // first real video stream (capital V skips cover art) + all audio + subtitles
        .args(["-map", "0:V:0", "-map", "0:a?", "-map", "0:s?"])
        .args(["-map_metadata", "0", "-map_chapters", "0"])
        .args(["-vf", &scale_filter(settings.max_res)])
        .args(["-filter_threads", "1"]);

    match (engine, settings.codec) {
        (Engine::Cpu, Codec::H265) => {
            let crf = settings.quality.crf(Codec::H265).to_string();
            cmd.args(["-c:v", "libx265", "-preset", "medium", "-crf", &crf])
                .args(["-x265-params", &format!("log-level=error:pools={threads}")]);
        }
        (Engine::Cpu, Codec::H264) => {
            let crf = settings.quality.crf(Codec::H264).to_string();
            cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", &crf])
                .args(["-threads", &threads_s]);
        }
        (Engine::Gpu, codec) => {
            let enc = if codec == Codec::H265 { "hevc_videotoolbox" } else { "h264_videotoolbox" };
            let kbps = vt_kbps(settings.max_res, codec, settings.quality);
            cmd.args(["-c:v", enc, "-allow_sw", "0"])
                .args(["-b:v", &format!("{kbps}k")])
                .args(["-maxrate", &format!("{}k", kbps * 3 / 2)])
                .args(["-bufsize", &format!("{}k", kbps * 2)]);
        }
    }
    if settings.codec == Codec::H265 && is_mp4 {
        cmd.args(["-tag:v", "hvc1"]); // lets QuickTime/Finder play it
    }
    cmd.args(["-c:a", "copy", "-c:s", "copy"]);
    if is_mp4 {
        // use_metadata_tags lets MP4 carry our custom `shrunk` key
        cmd.args(["-movflags", "+faststart+use_metadata_tags"]);
    }
    let mark = format!(
        "movie-shrinker {} {} {}",
        settings.max_res.label(),
        settings.codec.label().split(' ').next().unwrap_or(""),
        engine.label()
    );
    cmd.args(["-metadata", &format!("{TAG}={mark}")]);
    cmd.arg(&tmp)
        .stdin(if stream { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child: Child = cmd.spawn().map_err(|e| format!("ffmpeg: {e}"))?;
    // Duration comes from ffmpeg's own header, so each file is opened once
    // (a separate ffprobe re-scans the whole file: minutes on a busy HDD).
    let duration = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let stderr = child.stderr.take().unwrap();
    let dur = duration.clone();
    let err_thread = std::thread::spawn(move || {
        let mut errors = Vec::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if dur.load(Ordering::Relaxed) == 0 {
                if let Some(us) = parse_duration_us(&line) {
                    dur.store(us.max(1), Ordering::Relaxed);
                }
            }
            let l = line.to_ascii_lowercase();
            if l.contains("error") || l.contains("invalid") || l.contains("not found") || l.contains("failed") {
                errors.push(line);
            }
        }
        errors
    });

    let stdout = BufReader::new(child.stdout.take().unwrap());

    // Watchdog: honour Stop/Quit even while ffmpeg prints nothing (opening a
    // big recording on a busy USB disk can take minutes). Killing the child
    // closes its stdout, which ends the read loop below.
    let stdin = child.stdin.take();
    let child = std::sync::Mutex::new(child);
    let finished = AtomicBool::new(false);
    let speed = std::thread::scope(|scope| {
        if let Some(mut stdin) = stdin {
            let fed = &fed;
            scope.spawn(move || {
                let Ok(mut f) = std::fs::File::open(path) else { return };
                let mut buf = vec![0u8; FEED_CHUNK];
                while !cancel.load(Ordering::Relaxed) {
                    match f.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            // write fails (EPIPE) once ffmpeg exits or is killed
                            if stdin.write_all(&buf[..n]).is_err() {
                                break;
                            }
                            fed.fetch_add(n as u64, Ordering::Relaxed);
                        }
                    }
                }
                // dropping stdin sends EOF
            });
        }
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                if cancel.load(Ordering::Relaxed) {
                    let _ = child.lock().unwrap().kill();
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(150));
            }
        });
        let fraction = |out_us: u64| -> Option<f64> {
            if stream {
                // fragmented recordings often carry no usable duration; bytes are exact
                Some(fed.load(Ordering::Relaxed) as f64 / old_size.max(1) as f64)
            } else {
                let total = duration.load(Ordering::Relaxed);
                (total > 0).then(|| out_us as f64 / total as f64)
            }
        };
        let speed = read_progress(stdout, &fraction, &mut on_progress);
        finished.store(true, Ordering::Relaxed);
        speed
    });
    let status = loop {
        if let Some(st) = child.lock().unwrap().try_wait().map_err(|e| e.to_string())? {
            break st;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    if cancel.load(Ordering::Relaxed) {
        return Ok(Outcome::Cancelled);
    }
    if !status.success() {
        let errors = err_thread.join().unwrap_or_default();
        let first = errors.into_iter().find(|l| !l.trim().is_empty());
        return Err(first.unwrap_or_else(|| "encode failed".into()).trim().to_string());
    }
    on_progress(1.0, speed);

    let new_size = std::fs::metadata(&tmp).map_err(|e| e.to_string())?.len();
    if new_size == 0 {
        return Err("encoder produced an empty file".into());
    }
    if new_size >= old_size {
        // remember the verdict so the file isn't re-encoded for nothing next time
        set_xattr(path, "kept-original");
        return Ok(Outcome::KeptOriginal { old: old_size, new: new_size });
    }
    // Keep original permissions, stamp modification date = now, then atomically swap.
    let _ = std::fs::set_permissions(&tmp, meta.permissions());
    if let Ok(f) = std::fs::File::options().write(true).open(&tmp) {
        let _ = f.set_modified(SystemTime::now());
    }
    set_xattr(&tmp, &mark);
    std::fs::rename(&tmp, path).map_err(|e| format!("replace failed: {e}"))?;
    Ok(Outcome::Replaced { old: old_size, new: new_size })
}

/// Reads ffmpeg's `-progress` key=value stream until it closes; returns the last speed.
fn read_progress(
    stdout: BufReader<std::process::ChildStdout>,
    fraction: &dyn Fn(u64) -> Option<f64>,
    on_progress: &mut impl FnMut(f32, f32),
) -> f32 {
    let mut speed = 0.0f32;
    for line in stdout.lines().map_while(Result::ok) {
        if let Some(v) = line.strip_prefix("out_time_us=") {
            if let Ok(us) = v.trim().parse::<u64>() {
                if let Some(f) = fraction(us) {
                    on_progress(f.clamp(0.0, 1.0) as f32, speed);
                }
            }
        } else if let Some(v) = line.strip_prefix("speed=") {
            speed = v.trim().trim_end_matches('x').parse().unwrap_or(speed);
        }
    }
    speed
}

/// Sequential read size when streaming a file into ffmpeg.
const FEED_CHUNK: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    /// MP4/MOV with the `moov` index before the media (fast start).
    MoovFirst,
    /// Fragmented MP4 (`moov` then `moof` fragments) — typical for recordings.
    Fragmented,
    /// Classic MP4 with the index at the end: needs seeking.
    MoovLast,
    Matroska,
    Unknown,
}

impl Layout {
    /// ffmpeg can read it front to back, so it can be streamed in big chunks.
    fn streamable(self) -> bool {
        matches!(self, Layout::MoovFirst | Layout::Fragmented | Layout::Matroska)
    }
    /// ffprobe reads only the header (cheap even on a slow disk).
    fn cheap_to_probe(self) -> bool {
        matches!(self, Layout::MoovFirst | Layout::Matroska)
    }
}

/// Walks the top-level MP4 boxes (a few tiny reads at the front of the file).
fn layout(path: &Path) -> Layout {
    let ext = path.extension().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
    // never open FIFOs/devices: opening a FIFO blocks until a writer appears
    if !std::fs::metadata(path).map(|m| m.is_file()).unwrap_or(false) {
        return Layout::Unknown;
    }
    if ext == "mkv" {
        return Layout::Matroska;
    }
    let Ok(mut f) = std::fs::File::open(path) else { return Layout::Unknown };
    let mut pos = 0u64;
    let mut seen_moov = false;
    for _ in 0..32 {
        let mut hdr = [0u8; 16];
        if f.seek(SeekFrom::Start(pos)).is_err() || f.read_exact(&mut hdr[..8]).is_err() {
            break;
        }
        let mut size = u32::from_be_bytes(hdr[0..4].try_into().unwrap()) as u64;
        let kind: [u8; 4] = hdr[4..8].try_into().unwrap();
        if size == 1 {
            if f.read_exact(&mut hdr[8..16]).is_err() {
                break;
            }
            size = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
        }
        match &kind {
            b"moov" => seen_moov = true,
            b"moof" => return if seen_moov { Layout::Fragmented } else { Layout::Unknown },
            b"mdat" => return if seen_moov { Layout::MoovFirst } else { Layout::MoovLast },
            _ => {}
        }
        if size < 8 {
            break;
        }
        pos += size;
    }
    if seen_moov { Layout::MoovFirst } else { Layout::Unknown }
}

/// Extended attribute set on every file this tool has finished with.
const XATTR: &str = "com.movieshrinker.shrunk";
/// Metadata tag written into every shrunk file (travels with the file).
const TAG: &str = "shrunk";

fn has_xattr(path: &Path) -> bool {
    Command::new("/usr/bin/xattr")
        .args(["-p", XATTR])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn set_xattr(path: &Path, value: &str) {
    let _ = Command::new("/usr/bin/xattr").args(["-w", XATTR, value]).arg(path).status();
}

/// Already processed? Instant xattr check first; for files whose header is
/// cheap to read, also the in-file tag, or "already HEVC at/below target"
/// (catches files shrunk before tagging existed). Fragmented originals are
/// never ffprobed here: that would scan the whole recording.
fn already_shrunk(path: &Path, layout: Layout, res: MaxRes) -> bool {
    // tools below open the file; a FIFO/device would block them forever
    if !std::fs::metadata(path).map(|m| m.is_file()).unwrap_or(false) {
        return false;
    }
    if has_xattr(path) {
        return true;
    }
    if !layout.cheap_to_probe() {
        return false;
    }
    let Ok(probe) = ffprobe_path() else { return false };
    let Ok(out) = Command::new(probe)
        .args(["-v", "error", "-select_streams", "V:0"])
        .args(["-show_entries", &format!("format_tags={TAG}:stream=codec_name,width,height")])
        .args(["-of", "default=nw=1"])
        .arg(path)
        .output()
    else {
        return false;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(|v| v.trim().to_string())
    };
    if field(&format!("TAG:{TAG}")).is_some() {
        set_xattr(path, "tag");
        return true;
    }
    let (long, short) = res.bounds();
    let w: u32 = field("width").and_then(|v| v.parse().ok()).unwrap_or(u32::MAX);
    let h: u32 = field("height").and_then(|v| v.parse().ok()).unwrap_or(u32::MAX);
    let fits = w.max(h) <= long && w.min(h) <= short;
    if field("codec_name").as_deref() == Some("hevc") && fits {
        set_xattr(path, "hevc");
        return true;
    }
    false
}

fn ffprobe_path() -> Result<&'static Path, String> {
    static P: OnceLock<Option<PathBuf>> = OnceLock::new();
    P.get_or_init(|| find_tool("ffprobe"))
        .as_deref()
        .ok_or_else(|| "ffprobe not found".into())
}

/// Max parallel encodes reading from one spinning/external disk. More than
/// this makes the head seek between files and every encode drops to ~1×.
pub const MAX_PER_SLOW_DISK: usize = 2;

fn plist_true(plist: &str, key: &str) -> bool {
    plist
        .split(&format!("<key>{key}</key>"))
        .nth(1)
        .map(|rest| rest.trim_start().starts_with("<true/>"))
        .unwrap_or(false)
}

fn plist_string(plist: &str, key: &str) -> Option<String> {
    let rest = plist.split(&format!("<key>{key}</key>")).nth(1)?.trim_start();
    let v = rest.strip_prefix("<string>")?.split("</string>").next()?;
    Some(v.to_string())
}

/// Where a file lives: fast storage or not, the physical disk (as named by
/// `iostat`) and the volume name for display.
#[derive(Clone, Debug, Default)]
pub struct Storage {
    pub fast: bool,
    pub disk: String,
    pub volume: String,
}

/// Device id of the file's volume plus its storage info. Fast = SSD or
/// internal; USB enclosures often don't report SolidState, so anything
/// external without it counts as a hard disk.
pub fn storage_of(path: &Path) -> (u64, Storage) {
    use std::collections::HashMap;
    use std::os::unix::fs::MetadataExt;
    use std::sync::Mutex;
    static CACHE: OnceLock<Mutex<HashMap<u64, Storage>>> = OnceLock::new();

    let dev = std::fs::metadata(path).map(|m| m.dev()).unwrap_or(0);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(st) = cache.lock().unwrap().get(&dev) {
        return (dev, st.clone());
    }
    let st = (|| {
        let out = Command::new("/usr/bin/stat").args(["-f", "%Sd"]).arg(path).output().ok()?;
        let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let out = Command::new("/usr/sbin/diskutil").args(["info", "-plist", &id]).output().ok()?;
        let plist = String::from_utf8_lossy(&out.stdout);
        // APFS volumes live in a synthesized container; iostat reports the physical store
        let disk = plist_string(&plist, "APFSPhysicalStore")
            .or_else(|| plist_string(&plist, "ParentWholeDisk"))
            .map(|d| match d.rfind('s') {
                Some(i) if i > 4 && d[i + 1..].chars().all(|c| c.is_ascii_digit()) => d[..i].to_string(),
                _ => d,
            })
            .unwrap_or_default();
        Some(Storage {
            fast: plist_true(&plist, "SolidState") || plist_true(&plist, "Internal"),
            disk,
            volume: plist_string(&plist, "VolumeName").unwrap_or(id),
        })
    })()
    .unwrap_or_default();
    cache.lock().unwrap().insert(dev, st.clone());
    (dev, st)
}

/// Recursively collect video files from dropped paths (files or folders).
pub fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if let Ok(rd) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            for p in entries {
                let hidden = p.file_name().map(|n| n.to_string_lossy().starts_with('.')).unwrap_or(true);
                if !hidden {
                    collect(&p, out);
                }
            }
        }
    } else if is_video(path) {
        out.push(path.to_path_buf());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffmpeg_duration_line() {
        let l = "  Duration: 00:17:12.40, start: 0.000000, bitrate: 5177 kb/s";
        assert_eq!(parse_duration_us(l), Some(1_032_400_000));
        assert_eq!(parse_duration_us("  Duration: N/A, start: 0.0"), None);
        assert_eq!(parse_duration_us("Stream #0:0: Video: h264"), None);
    }

    #[test]
    fn vt_budget_scales_with_resolution_and_quality() {
        assert_eq!(vt_kbps(MaxRes::P720, Codec::H265, Quality::Balanced), 1600);
        assert!(vt_kbps(MaxRes::P1440, Codec::H265, Quality::Small) > 1600);
        assert!(vt_kbps(MaxRes::P720, Codec::H264, Quality::High) > vt_kbps(MaxRes::P720, Codec::H265, Quality::High));
    }

    /// ffmpeg blocked opening its input prints nothing; Stop must still kill it.
    #[test]
    fn cancel_kills_silent_ffmpeg() {
        let dir = std::env::temp_dir().join(format!("shrink-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("hang.mp4");
        let _ = std::fs::remove_file(&fifo);
        assert!(Command::new("mkfifo").arg(&fifo).status().unwrap().success());

        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let c = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(1));
            c.store(true, Ordering::Relaxed);
        });
        let t = std::time::Instant::now();
        let r = shrink(&fifo, Settings::default(), Engine::Gpu, &cancel, |_, _| {});
        assert!(matches!(r, Ok(Outcome::Cancelled)), "expected Cancelled");
        assert!(t.elapsed() < std::time::Duration::from_secs(3), "took {:?}", t.elapsed());
        assert!(!temp_path(&fifo).exists(), "temp file left behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn make(dir: &Path, name: &str, extra: &[&str]) -> PathBuf {
        let out = dir.join(name);
        let st = Command::new(ffmpeg().unwrap())
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc2=size=1920x1080:rate=30:duration=6")
            .args(["-f", "lavfi", "-i", "sine=duration=6", "-c:v", "libx264", "-preset", "ultrafast"])
            .args(["-c:a", "aac", "-shortest"])
            .args(extra)
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success());
        out
    }

    fn duration_s(p: &Path) -> f64 {
        let o = Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"])
            .arg(p)
            .output()
            .unwrap();
        String::from_utf8_lossy(&o.stdout).trim().parse().unwrap()
    }

    /// Streamed (piped) and seek-mode inputs both encode completely.
    #[test]
    fn streams_when_possible_and_keeps_full_duration() {
        let dir = std::env::temp_dir().join(format!("shrink-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cases = [
            (make(&dir, "frag.mp4", &["-movflags", "frag_keyframe+empty_moov"]), true),
            (make(&dir, "fast.mp4", &["-movflags", "+faststart"]), true),
            (make(&dir, "tail.mp4", &[]), false), // moov at the end
            (make(&dir, "m.mkv", &[]), true),
        ];
        for (path, expect_stream) in &cases {
            assert_eq!(layout(path).streamable(), *expect_stream, "{}", path.display());
            let mut last = 0.0f32;
            let r = shrink(path, Settings::default(), Engine::Gpu, &AtomicBool::new(false), |p, _| last = p);
            assert!(matches!(r, Ok(Outcome::Replaced { .. })), "{}: not replaced", path.display());
            let d = duration_s(path);
            assert!((d - 6.0).abs() < 0.2, "{}: duration {d}", path.display());
            assert!(last > 0.9, "{}: progress only reached {last}", path.display());
            // second run must skip instantly via the mark
            let again = shrink(path, Settings::default(), Engine::Gpu, &AtomicBool::new(false), |_, _| {});
            assert!(matches!(again, Ok(Outcome::AlreadyShrunk)), "{}: re-encoded", path.display());
            // and the in-file tag alone is enough (e.g. after copying without xattrs)
            Command::new("/usr/bin/xattr").args(["-d", XATTR]).arg(path).status().unwrap();
            assert!(already_shrunk(path, layout(path), MaxRes::P720), "{}: tag not found", path.display());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
