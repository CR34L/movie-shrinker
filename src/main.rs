#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod encode;
mod stats;

use eframe::egui;
use encode::{Codec, Engine, MaxRes, Mode, Outcome, Quality, Settings};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
enum Status {
    Queued,
    Running { p: f32, speed: f32 },
    Done { old: u64, new: u64 },
    Kept { old: u64, new: u64 },
    Failed(String),
    /// Already shrunk earlier — skipped.
    Skipped,
    Cancelled,
}

impl Status {
    fn active(&self) -> bool {
        matches!(self, Status::Queued | Status::Running { .. })
    }
}

struct Job {
    path: PathBuf,
    size: u64,
    /// Volume device id and whether it is SSD/internal.
    dev: u64,
    fast_disk: bool,
    /// physical disk (iostat name) and volume name, for the footer
    disk: String,
    volume: String,
    settings: Settings,
    status: Status,
    engine: Option<Engine>,
    /// Set after a hardware encode failed, so only CPU lanes retry it.
    no_gpu: bool,
    cancel: Arc<AtomicBool>,
}

impl Job {
    fn allows(&self, engine: Engine) -> bool {
        match engine {
            Engine::Gpu => self.settings.mode != Mode::CpuOnly && !self.no_gpu,
            Engine::Cpu => self.settings.mode != Mode::GpuOnly || self.no_gpu,
        }
    }
}

type Notify = Arc<dyn Fn() + Send + Sync>;

/// Shared job queue served by parallel encode lanes: some on the hardware
/// encoder(s), some on CPU cores. GPU lanes take the biggest files, CPU
/// lanes the smallest, so the slow lanes don't end up holding the long film.
#[derive(Default)]
struct Queue {
    jobs: Vec<Job>,
    cpu_lanes: usize,
    gpu_lanes: usize,
}

#[derive(Clone)]
struct Pool {
    q: Arc<Mutex<Queue>>,
    notify: Notify,
}

impl Pool {
    fn new(notify: Notify) -> Self {
        Self { q: Default::default(), notify }
    }

    /// Scans on a background thread: a busy external disk must not freeze the UI.
    fn add(&self, paths: Vec<PathBuf>, settings: Settings) -> std::thread::JoinHandle<()> {
        let pool = self.clone();
        std::thread::spawn(move || pool.add_blocking(&paths, settings))
    }

    fn add_blocking(&self, paths: &[PathBuf], settings: Settings) {
        let mut files = Vec::new();
        for p in paths {
            encode::collect(p, &mut files);
        }
        let files: Vec<_> = files
            .into_iter()
            .map(|path| {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let (dev, st) = encode::storage_of(&path);
                (path, size, dev, st)
            })
            .collect();
        {
            let mut q = self.q.lock().unwrap();
            for (path, size, dev, st) in files {
                if q.jobs.iter().any(|j| j.path == path && j.status.active()) {
                    continue;
                }
                q.jobs.push(Job {
                    path,
                    size,
                    dev,
                    fast_disk: st.fast,
                    disk: st.disk,
                    volume: st.volume,
                    settings,
                    status: Status::Queued,
                    engine: None,
                    no_gpu: false,
                    cancel: Default::default(),
                });
            }
        }
        self.ensure_lanes();
    }

    /// Start as many lanes per engine as the queued jobs can use, respecting
    /// the per-disk limit on hard disks. GPU lanes start first: on a slow
    /// disk the few read slots should go to the fastest encoder.
    fn ensure_lanes(&self) {
        let mut q = self.q.lock().unwrap();
        let mut gpu_started = 0;
        for engine in [Engine::Gpu, Engine::Cpu] {
            let want = q
                .jobs
                .iter()
                .filter(|j| matches!(j.status, Status::Queued) && j.allows(engine))
                .map(|j| {
                    let (c, g) = if j.no_gpu { (j.settings.mode.lanes().0.max(1), 0) } else { j.settings.mode.lanes() };
                    if engine == Engine::Cpu { c } else { g }
                })
                .max()
                .unwrap_or(0);
            let running = if engine == Engine::Cpu { q.cpu_lanes } else { q.gpu_lanes };
            // slots just handed to GPU lanes aren't marked Running yet
            let free = startable(&q.jobs, engine).saturating_sub(if engine == Engine::Cpu { gpu_started } else { 0 });
            let to_start = want.saturating_sub(running).min(free);
            if engine == Engine::Cpu {
                q.cpu_lanes += to_start;
            } else {
                q.gpu_lanes += to_start;
                gpu_started = to_start;
            }
            for _ in 0..to_start {
                let pool = self.clone();
                std::thread::spawn(move || pool.lane(engine));
            }
        }
    }

    fn lane(&self, engine: Engine) {
        loop {
            let next = {
                let mut q = self.q.lock().unwrap();
                let candidates = q
                    .jobs
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| eligible(&q.jobs, j, engine));
                let pick = match engine {
                    Engine::Gpu => candidates.max_by_key(|(_, j)| j.size),
                    Engine::Cpu => candidates.min_by_key(|(_, j)| j.size),
                }
                .map(|(i, _)| i);
                match pick {
                    Some(i) => {
                        let j = &mut q.jobs[i];
                        j.status = Status::Running { p: 0.0, speed: 0.0 };
                        j.engine = Some(engine);
                        Some((i, j.path.clone(), j.settings, j.cancel.clone()))
                    }
                    None => {
                        match engine {
                            Engine::Cpu => q.cpu_lanes -= 1,
                            Engine::Gpu => q.gpu_lanes -= 1,
                        }
                        None
                    }
                }
            };
            let Some((i, path, settings, cancel)) = next else { break };
            (self.notify)();

            let set = |st: Status| {
                self.q.lock().unwrap().jobs[i].status = st;
                (self.notify)();
            };
            let result = encode::shrink(&path, settings, engine, &cancel, |p, speed| {
                set(Status::Running { p, speed })
            });
            match result {
                Ok(Outcome::Replaced { old, new }) => set(Status::Done { old, new }),
                Ok(Outcome::KeptOriginal { old, new }) => set(Status::Kept { old, new }),
                Ok(Outcome::AlreadyShrunk) => set(Status::Skipped),
                Ok(Outcome::Cancelled) => set(Status::Cancelled),
                // Hardware encoder unavailable/failed: hand it to the CPU.
                Err(_) if engine == Engine::Gpu && !cancel.load(Ordering::Relaxed) => {
                    {
                        let mut q = self.q.lock().unwrap();
                        q.jobs[i].no_gpu = true;
                        q.jobs[i].status = Status::Queued;
                    }
                    self.ensure_lanes();
                }
                Err(e) => set(Status::Failed(e)),
            }
            self.ensure_lanes();
        }
        (self.notify)();
    }

    fn cancel_all(&self) {
        let mut q = self.q.lock().unwrap();
        for j in q.jobs.iter_mut() {
            match j.status {
                Status::Queued => j.status = Status::Cancelled,
                Status::Running { .. } => j.cancel.store(true, Ordering::Relaxed),
                _ => {}
            }
        }
    }

    fn busy(&self) -> bool {
        let q = self.q.lock().unwrap();
        q.cpu_lanes + q.gpu_lanes > 0
    }
}

fn running_on(jobs: &[Job], dev: u64) -> usize {
    jobs.iter().filter(|j| j.dev == dev && matches!(j.status, Status::Running { .. })).count()
}

/// Queued, allowed on this engine, and its disk has a free read slot.
fn eligible(jobs: &[Job], j: &Job, engine: Engine) -> bool {
    matches!(j.status, Status::Queued)
        && j.allows(engine)
        && (j.fast_disk || running_on(jobs, j.dev) < encode::MAX_PER_SLOW_DISK)
}

/// How many more jobs could start right now on this engine.
fn startable(jobs: &[Job], engine: Engine) -> usize {
    let mut slow: std::collections::HashMap<u64, usize> = Default::default();
    let mut n = 0;
    for j in jobs.iter().filter(|j| eligible(jobs, j, engine)) {
        if j.fast_disk {
            n += 1;
        } else {
            *slow.entry(j.dev).or_default() += 1;
        }
    }
    for (dev, queued) in slow {
        n += queued.min(encode::MAX_PER_SLOW_DISK.saturating_sub(running_on(jobs, dev)));
    }
    n
}

/// "Downloads  R 67 / W 3 MB/s · CPU 72% · GPU 15% · ENC 2/2 · 0 CPU + 2 GPU lanes · total 9.4×"
fn footer(q: &Queue, st: &stats::Stats, busy: bool) -> String {
    let mut parts = Vec::new();
    // read/write speed of the drives holding the files being processed
    let mut seen = std::collections::HashSet::new();
    let running = q.jobs.iter().filter(|j| matches!(j.status, Status::Running { .. }));
    let queued = q.jobs.iter().filter(|j| matches!(j.status, Status::Queued));
    for j in running.chain(queued) {
        if seen.insert(j.disk.clone()) {
            if let Some((r, w)) = st.disk_mbs.get(&j.disk) {
                parts.push(format!("{}  R {r:.0} / W {w:.0} MB/s", j.volume));
            }
        }
    }
    if parts.is_empty() {
        let (r, w) = st.disk_mbs.values().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        parts.push(format!("Disk  R {r:.0} / W {w:.0} MB/s"));
    }
    parts.push(format!("CPU {:.0}%", st.cpu_busy));
    parts.push(format!("GPU {:.0}%", st.gpu_busy));
    let (_, hw) = Mode::All.lanes();
    let enc = q
        .jobs
        .iter()
        .filter(|j| matches!(j.status, Status::Running { .. }) && j.engine == Some(Engine::Gpu))
        .count();
    parts.push(format!("ENC {enc}/{hw}"));
    if busy {
        let total: f32 = q
            .jobs
            .iter()
            .map(|j| if let Status::Running { speed, .. } = j.status { speed } else { 0.0 })
            .sum();
        parts.push(format!("{} CPU + {} GPU lanes", q.cpu_lanes, q.gpu_lanes));
        parts.push(format!("total {total:.1}×"));
    }
    parts.join("  ·  ")
}

fn mb(b: u64) -> String {
    let m = b as f64 / 1_048_576.0;
    if m >= 1024.0 { format!("{:.2} GB", m / 1024.0) } else { format!("{m:.1} MB") }
}

struct App {
    settings: Settings,
    pool: Pool,
    stats: Arc<Mutex<stats::Stats>>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();
        let ctx = cc.egui_ctx.clone();
        Self {
            settings,
            pool: Pool::new(Arc::new(move || ctx.request_repaint())),
            stats: stats::start(),
        }
    }
}

fn combo<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    options: &[T],
    label: fn(T) -> &'static str,
) {
    egui::ComboBox::from_id_salt(id).selected_text(label(*value)).show_ui(ui, |ui| {
        for &o in options {
            ui.selectable_value(value, o, label(o));
        }
    });
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }

    fn on_exit(&mut self) {
        // Kill running encodes; each lane deletes its temp file on the way out.
        self.pool.cancel_all();
        for _ in 0..50 {
            if !self.pool.busy() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
        let (dropped, hovering) = ctx.input(|i| {
            (
                i.raw
                    .dropped_files
                    .iter()
                    .map(|f| f.path().to_path_buf())
                    .filter(|p| !p.as_os_str().is_empty())
                    .collect::<Vec<_>>(),
                !i.raw.hovered_files.is_empty(),
            )
        });
        if !dropped.is_empty() {
            self.pool.add(dropped, self.settings);
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let s = &mut self.settings;
                combo(ui, "res", &mut s.max_res, &[MaxRes::P720, MaxRes::P1440], MaxRes::label);
                combo(ui, "codec", &mut s.codec, &[Codec::H265, Codec::H264], Codec::label);
                combo(
                    ui,
                    "quality",
                    &mut s.quality,
                    &[Quality::High, Quality::Balanced, Quality::Small],
                    Quality::label,
                );
                combo(ui, "mode", &mut s.mode, &[Mode::All, Mode::CpuOnly, Mode::GpuOnly], Mode::label);
            });
            ui.add_space(6.0);

            // drop zone
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(ui.available_width(), 70.0), egui::Sense::hover());
            let vis = ui.visuals();
            let stroke_col = if hovering { vis.selection.bg_fill } else { vis.weak_text_color() };
            ui.painter().rect_stroke(
                rect,
                8.0,
                egui::Stroke::new(if hovering { 2.5 } else { 1.0 }, stroke_col),
                egui::StrokeKind::Inside,
            );
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                if hovering { "Release to shrink" } else { "Drop movies here  (mp4 · mkv · mov · m4v)" },
                egui::FontId::proportional(15.0),
                vis.text_color(),
            );
            ui.add_space(6.0);

            if encode::ffmpeg().is_err() {
                ui.colored_label(ui.visuals().error_fg_color, "ffmpeg not found — run: brew install ffmpeg");
            }

            let busy = self.pool.busy();
            let mut q = self.pool.q.lock().unwrap();

            ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
                ui.horizontal(|ui| {
                    if q.jobs.iter().any(|j| !j.status.active()) && ui.button("Clear").clicked() {
                        q.jobs.retain(|j| j.status.active());
                    }
                    if busy && ui.button("Stop all").clicked() {
                        for j in q.jobs.iter_mut() {
                            match j.status {
                                Status::Queued => j.status = Status::Cancelled,
                                Status::Running { .. } => j.cancel.store(true, Ordering::Relaxed),
                                _ => {}
                            }
                        }
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.weak(footer(&q, &self.stats.lock().unwrap(), busy));
                    });
                });

                ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        for job in q.jobs.iter_mut().rev() {
                            let name = job.path.file_name().unwrap_or_default().to_string_lossy();
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new(name).strong())
                                    .on_hover_text(job.path.display().to_string());
                                if job.status.active() {
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.small_button("×").on_hover_text("Cancel this file").clicked() {
                                            if matches!(job.status, Status::Queued) {
                                                job.status = Status::Cancelled;
                                            } else {
                                                job.cancel.store(true, Ordering::Relaxed);
                                            }
                                        }
                                    });
                                }
                            });
                            let engine = job.engine.map(|e| e.label()).unwrap_or("");
                            match &job.status {
                                Status::Queued => {
                                    ui.weak("waiting…");
                                }
                                Status::Running { p, speed } => {
                                    let mut text = format!("{engine}  {:.0}%", p * 100.0);
                                    if *speed > 0.0 {
                                        text += &format!("  ·  {speed:.1}×");
                                    }
                                    ui.add(egui::ProgressBar::new(*p).text(text).animate(true));
                                }
                                Status::Done { old, new } => {
                                    let pct = 100.0 - (*new as f64 / *old as f64) * 100.0;
                                    ui.label(format!("✔ {} → {}  (−{pct:.0}%, {engine})", mb(*old), mb(*new)));
                                }
                                Status::Kept { old, new } => {
                                    ui.weak(format!(
                                        "Already small — kept original ({} vs {})",
                                        mb(*old),
                                        mb(*new)
                                    ));
                                }
                                Status::Failed(e) => {
                                    ui.colored_label(ui.visuals().error_fg_color, format!("✖ {e}"));
                                }
                                Status::Skipped => {
                                    ui.weak("already shrunk — skipped");
                                }
                                Status::Cancelled => {
                                    ui.weak("cancelled — original untouched");
                                }
                            }
                            ui.add_space(4.0);
                        }
                    });
                });
            });
        });
    }
}

/// Headless mode: movie-shrinker --shrink [--2k] [--h264] [--cpu|--gpu] FILE...
/// Uses the same parallel lane pool as the GUI.
fn cli(args: Vec<String>) -> i32 {
    let mut s = Settings::default();
    let mut paths = Vec::new();
    for a in args {
        match a.as_str() {
            "--2k" => s.max_res = MaxRes::P1440,
            "--h264" => s.codec = Codec::H264,
            "--cpu" => s.mode = Mode::CpuOnly,
            "--gpu" => s.mode = Mode::GpuOnly,
            _ => paths.push(PathBuf::from(a)),
        }
    }
    let (c, g) = s.mode.lanes();
    eprintln!("lanes: {c} CPU + {g} GPU");
    let pool = Pool::new(Arc::new(|| {}));
    pool.add(paths, s).join().ok();
    while pool.busy() {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let mut code = 0;
    for j in &pool.q.lock().unwrap().jobs {
        let e = j.engine.map(|e| e.label()).unwrap_or("-");
        let msg = match &j.status {
            Status::Done { old, new } => format!("{} -> {}", mb(*old), mb(*new)),
            Status::Kept { .. } => "kept original (not smaller)".into(),
            Status::Skipped => "already shrunk — skipped".into(),
            Status::Failed(err) => {
                code = 1;
                format!("ERROR: {err}")
            }
            _ => "cancelled".into(),
        };
        eprintln!("[{e}] {}  {msg}", j.path.display());
    }
    code
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|a| a == "--shrink").unwrap_or(false) {
        std::process::exit(cli(args[1..].to_vec()));
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Movie Shrinker")
            .with_inner_size([520.0, 360.0])
            .with_min_inner_size([380.0, 240.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Movie Shrinker", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
