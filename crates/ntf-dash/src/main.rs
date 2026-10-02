//! ntf-dash: a fullscreen terminal dashboard for NeuralType training
//! runs. Reads the trainer's log, polls the GPU, and draws the whole
//! story: an IoU chart across every epoch, the live numbers, recent
//! epochs, and the hardware. House palette: ink green, gold, red.
//!
//! Usage: ntf-dash [log-path] [total-epochs]
//! Defaults: the newest data/train-*.log under the cwd, and the
//! epoch count the log states (60 for logs from before it did).
//! Keys: Tab switches to the next run's log, q quits.

use std::io;
use std::process::Command;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table};

const INK: Color = Color::Rgb(42, 163, 95);
const GOLD: Color = Color::Rgb(201, 162, 39);
const RED: Color = Color::Rgb(239, 68, 68);
const GRAY: Color = Color::Rgb(138, 138, 138);
const DIM: Color = Color::Rgb(77, 77, 77);

#[derive(Clone, Default)]
struct Epoch {
    n: u32,
    loss: f64,
    mse: f64,
    iou: f64,
    secs: f64,
    /// Fine-tuning extras, when the run reports them.
    labeled: Option<f64>,
    disp_err: Option<f64>,
    stretch: Option<f64>,
}

#[derive(Default)]
struct RunInfo {
    device: String,
    rows: String,
    model: String,
    lr: String,
    batch: String,
    oversample: String,
    resumed: bool,
    /// What the run trains on beyond the teacher rows.
    extra: String,
    /// Epochs the log says it will run, summed over resumed legs.
    planned: u32,
}

#[derive(Default, Clone)]
struct Gpu {
    util: u16,
    mem_used: f64,
    mem_total: f64,
    temp: u16,
    watts: f64,
    ok: bool,
}

fn parse_log(path: &str) -> (RunInfo, Vec<Epoch>) {
    let mut info = RunInfo::default();
    let mut epochs: Vec<Epoch> = Vec::new();
    let mut epoch_offset: u32 = 0;
    let Ok(text) = std::fs::read_to_string(path) else {
        return (info, epochs);
    };
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if line.starts_with("device:") {
            info.device = line["device:".len()..].trim().to_string();
        } else if line.starts_with("rows:") {
            info.rows = line["rows:".len()..].trim().to_string();
        } else if line.starts_with("model:") {
            info.model = line["model:".len()..].trim().to_string();
        } else if line.starts_with("lr:") {
            info.lr = line["lr:".len()..].trim().to_string();
        } else if line.starts_with("batch size:") {
            info.batch = line["batch size:".len()..].trim().to_string();
        } else if line.starts_with("oversampling") {
            info.oversample = line["oversampling".len()..].trim().to_string();
        } else if line.starts_with("resumed from") {
            info.resumed = true;
        } else if line.starts_with("epochs:") {
            info.planned += line["epochs:".len()..].trim().parse::<u32>().unwrap_or(0);
        } else if line.starts_with("stretch:")
            || line.starts_with("labeled rows")
            || line.starts_with("training the stretch adapter")
        {
            info.extra = line.trim().to_string();
        } else if t.len() >= 12 && t[0] == "epoch" {
            // epoch N train loss X val mse Y val IoU Z [labeled IoU A
            // disp err Bu] [stretch IoU C] (Ss)
            // Resumed legs restart the counter at 1 in the same log;
            // offset so the chart shows cumulative epochs.
            let secs = t[t.len() - 1].trim_matches(|c| c == '(' || c == ')' || c == 's');
            let after = |a: &str, b: &str| -> Option<f64> {
                let i = t.windows(2).position(|w| w[0] == a && w[1] == b)?;
                t.get(i + 2)?.trim_end_matches('u').parse().ok()
            };
            let raw: u32 = t[1].parse().unwrap_or(0);
            let last = epochs.last().map(|e: &Epoch| e.n).unwrap_or(0);
            if raw + epoch_offset <= last {
                epoch_offset = last;
            }
            epochs.push(Epoch {
                n: raw + epoch_offset,
                loss: t[4].parse().unwrap_or(0.0),
                mse: t[7].parse().unwrap_or(0.0),
                iou: t[10].parse().unwrap_or(0.0),
                secs: secs.parse().unwrap_or(0.0),
                labeled: after("labeled", "IoU"),
                disp_err: after("disp", "err"),
                stretch: after("stretch", "IoU"),
            });
        }
    }
    (info, epochs)
}

fn poll_gpu() -> Gpu {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else { return Gpu::default() };
    let s = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = s.trim().split(',').map(|x| x.trim()).collect();
    if f.len() < 5 {
        return Gpu::default();
    }
    Gpu {
        util: f[0].parse().unwrap_or(0),
        mem_used: f[1].parse().unwrap_or(0.0),
        mem_total: f[2].parse().unwrap_or(0.0),
        temp: f[3].parse().unwrap_or(0),
        watts: f[4].parse().unwrap_or(0.0),
        ok: true,
    }
}

/// The training logs under data/, newest first.
fn logs() -> Vec<String> {
    let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
    for e in std::fs::read_dir("data").into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("train-") && name.ends_with(".log") {
            if let Ok(m) = e.metadata().and_then(|m| m.modified()) {
                found.push((m, format!("data/{name}")));
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, p)| p).collect()
}

fn newest_log() -> Option<String> {
    logs().into_iter().next()
}

fn fmt_eta(secs: f64) -> String {
    let s = secs as u64;
    if s >= 3600 {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    } else {
        format!("{}m {:02}s", s / 60, s % 60)
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut log = args
        .get(1)
        .cloned()
        .or_else(newest_log)
        .unwrap_or_else(|| "data/train.log".into());
    let total_arg: Option<u32> = args.get(2).and_then(|s| s.parse().ok());

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut gpu = poll_gpu();
    let mut last_gpu = Instant::now();
    // test hook: render N frames, then exit
    let max_frames: Option<u64> =
        std::env::var("NTF_DASH_MAX_FRAMES").ok().and_then(|v| v.parse().ok());
    let mut frames = 0u64;

    loop {
        if last_gpu.elapsed() > Duration::from_secs(2) {
            gpu = poll_gpu();
            last_gpu = Instant::now();
        }
        let (info, epochs) = parse_log(&log);
        let total = total_arg.unwrap_or(if info.planned > 0 { info.planned } else { 60 });

        terminal.draw(|f| draw(f, &log, total, &info, &epochs, &gpu))?;

        frames += 1;
        if max_frames.is_some_and(|m| frames >= m) {
            break;
        }
        if event::poll(Duration::from_millis(700))? {
            if let Event::Key(k) = event::read()? {
                if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
                if k.code == KeyCode::Tab {
                    // next run: the logs in order, wrapping around
                    let all = logs();
                    if !all.is_empty() {
                        let at = all.iter().position(|p| *p == log).map_or(0, |i| i + 1);
                        log = all[at % all.len()].clone();
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn draw(f: &mut Frame, log: &str, total: u32, info: &RunInfo, epochs: &[Epoch], gpu: &Gpu) {
    let area = f.area();
    let rows = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(9),
        Constraint::Length(1),
    ])
    .split(area);

    // ── header ──────────────────────────────────────────────────
    let title = Line::from(vec![
        Span::styled(" ◉ NEURALTYPE ", Style::new().fg(INK).bold()),
        Span::styled("TRAINER ", Style::new().fg(GRAY).bold()),
        Span::styled(log, Style::new().fg(DIM)),
    ]);
    let sub = Line::from(vec![
        Span::styled(format!(" {} ", info.device), Style::new().fg(GOLD)),
        Span::styled(format!("· {} ", info.model), Style::new().fg(GRAY)),
        Span::styled(format!("· lr {} ", info.lr), Style::new().fg(GRAY)),
        Span::styled(format!("· batch {} ", info.batch), Style::new().fg(GRAY)),
        Span::styled(
            if info.resumed { "· resumed " } else { "" },
            Style::new().fg(GOLD),
        ),
    ]);
    let sub2 = Line::from(vec![
        Span::styled(format!(" rows {} ", info.rows), Style::new().fg(DIM)),
        Span::styled(format!("· oversampling {} ", info.oversample), Style::new().fg(DIM)),
        Span::styled(format!("· {} ", info.extra), Style::new().fg(DIM)),
    ]);
    f.render_widget(
        Paragraph::new(vec![title, sub, sub2])
            .block(Block::new().borders(Borders::BOTTOM).border_style(DIM)),
        rows[0],
    );

    // ── main: IoU chart + live numbers ──────────────────────────
    let mid = Layout::horizontal([Constraint::Percentage(66), Constraint::Percentage(34)])
        .split(rows[1]);

    let pts: Vec<(f64, f64)> = epochs.iter().map(|e| (e.n as f64, e.iou)).collect();
    let series = |get: fn(&Epoch) -> Option<f64>| -> Vec<(f64, f64)> {
        epochs.iter().filter_map(|e| get(e).map(|v| (e.n as f64, v))).collect()
    };
    let stretch_pts = series(|e| e.stretch);
    let labeled_pts = series(|e| e.labeled);
    let (ymin, ymax) = pts
        .iter()
        .chain(&stretch_pts)
        .chain(&labeled_pts)
        .fold((1.0f64, 0.0f64), |(lo, hi), &(_, y)| (lo.min(y), hi.max(y)));
    let ymin = (ymin - 0.02).max(0.0);
    let ymax = (ymax + 0.02).min(1.0);
    let xmax = epochs.last().map(|e| e.n as f64).unwrap_or(1.0).max(total as f64);
    let ds = Dataset::default()
        .name("val IoU")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(INK))
        .data(&pts);
    let mut sets = vec![ds];
    if !stretch_pts.is_empty() {
        sets.push(
            Dataset::default()
                .name("stretch IoU")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(GOLD))
                .data(&stretch_pts),
        );
    }
    if !labeled_pts.is_empty() {
        sets.push(
            Dataset::default()
                .name("labeled IoU")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(RED))
                .data(&labeled_pts),
        );
    }
    let chart = Chart::new(sets)
        .block(
            Block::bordered()
                .border_style(DIM)
                .title(Span::styled(" contour IoU per epoch ", Style::new().fg(GRAY))),
        )
        .x_axis(
            Axis::default()
                .bounds([0.0, xmax])
                .labels(["0".to_string(), format!("{xmax:.0}")])
                .style(Style::new().fg(DIM)),
        )
        .y_axis(
            Axis::default()
                .bounds([ymin, ymax])
                .labels([format!("{ymin:.2}"), format!("{ymax:.2}")])
                .style(Style::new().fg(DIM)),
        );
    f.render_widget(chart, mid[0]);

    let cur = epochs.last().cloned().unwrap_or_default();
    let best = epochs.iter().map(|e| e.iou).fold(0.0f64, f64::max);
    let avg_secs = if epochs.is_empty() {
        0.0
    } else {
        epochs.iter().map(|e| e.secs).sum::<f64>() / epochs.len() as f64
    };
    let remaining = (total.saturating_sub(cur.n)) as f64 * avg_secs;
    let pct = (cur.n as f64 / total as f64).clamp(0.0, 1.0);

    let right = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Min(3),
    ])
    .split(mid[1]);
    f.render_widget(
        Gauge::default()
            .block(Block::bordered().border_style(DIM).title(Span::styled(
                format!(" epoch {} / {total} ", cur.n),
                Style::new().fg(GRAY),
            )))
            .gauge_style(Style::new().fg(INK).bg(Color::Rgb(20, 20, 20)))
            .ratio(pct)
            .label(Span::styled(format!("{:.0}%", pct * 100.0), Style::new().fg(GOLD).bold())),
        right[0],
    );
    let extra = |label: &'static str, v: Option<f64>, digits: usize, color: Color| {
        v.map(|v| {
            Line::from(vec![
                Span::styled(label, Style::new().fg(GRAY)),
                Span::styled(format!("{v:.digits$}"), Style::new().fg(color).bold()),
            ])
        })
    };
    let mut stats = vec![
        Line::from(vec![
            Span::styled("   IoU  ", Style::new().fg(GRAY)),
            Span::styled(format!("{:.4}", cur.iou), Style::new().fg(INK).bold()),
        ]),
        Line::from(vec![
            Span::styled("  best  ", Style::new().fg(GRAY)),
            Span::styled(format!("{best:.4}"), Style::new().fg(GOLD)),
        ]),
        Line::from(vec![
            Span::styled("  loss  ", Style::new().fg(GRAY)),
            Span::styled(format!("{:.5}", cur.loss), Style::new().fg(RED)),
        ]),
        Line::from(vec![
            Span::styled(" epoch  ", Style::new().fg(GRAY)),
            Span::styled(format!("{:.0}s avg", avg_secs), Style::new().fg(GRAY)),
        ]),
        Line::from(vec![
            Span::styled("   eta  ", Style::new().fg(GRAY)),
            Span::styled(fmt_eta(remaining), Style::new().fg(GOLD).bold()),
        ]),
    ];
    stats.extend(extra("stretch ", cur.stretch, 4, GOLD));
    stats.extend(extra("labeled ", cur.labeled, 4, RED));
    stats.extend(extra("disp err", cur.disp_err, 1, GRAY));
    f.render_widget(
        Paragraph::new(stats).block(Block::bordered().border_style(DIM).title(Span::styled(
            " live ",
            Style::new().fg(GRAY),
        ))),
        right[1],
    );
    // loss sparkline over recent epochs
    let recent: Vec<u64> = epochs
        .iter()
        .rev()
        .take(right[2].width.saturating_sub(2) as usize)
        .map(|e| (e.loss * 1e6) as u64)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    f.render_widget(
        ratatui::widgets::Sparkline::default()
            .block(Block::bordered().border_style(DIM).title(Span::styled(
                " train loss ",
                Style::new().fg(GRAY),
            )))
            .style(Style::new().fg(RED))
            .data(&recent),
        right[2],
    );

    // ── bottom: recent epochs + GPU ─────────────────────────────
    let bot = Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(rows[2]);
    let tbl_rows: Vec<Row> = epochs
        .iter()
        .rev()
        .take(6)
        .map(|e| {
            Row::new(vec![
                format!("{}", e.n),
                format!("{:.5}", e.loss),
                format!("{:.5}", e.mse),
                format!("{:.4}", e.iou),
                e.stretch.map_or("-".into(), |v| format!("{v:.4}")),
                e.labeled.map_or("-".into(), |v| format!("{v:.4}")),
                format!("{:.0}s", e.secs),
            ])
            .style(Style::new().fg(GRAY))
        })
        .collect();
    f.render_widget(
        Table::new(
            tbl_rows,
            [
                Constraint::Length(6),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(6),
            ],
        )
        .header(
            Row::new(vec!["epoch", "loss", "val mse", "IoU", "stretch", "labeled", "s"])
                .style(Style::new().fg(GOLD).bold()),
        )
        .block(Block::bordered().border_style(DIM).title(Span::styled(
            " recent epochs ",
            Style::new().fg(GRAY),
        ))),
        bot[0],
    );

    let g = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(1),
    ])
    .split(bot[1].inner(Margin::new(0, 0)));
    if gpu.ok {
        f.render_widget(
            Gauge::default()
                .block(Block::bordered().border_style(DIM).title(Span::styled(
                    " GPU utilization ",
                    Style::new().fg(GRAY),
                )))
                .gauge_style(Style::new().fg(INK).bg(Color::Rgb(20, 20, 20)))
                .percent(gpu.util),
            g[0],
        );
        let vram = if gpu.mem_total > 0.0 { gpu.mem_used / gpu.mem_total } else { 0.0 };
        f.render_widget(
            Gauge::default()
                .block(Block::bordered().border_style(DIM).title(Span::styled(
                    " VRAM ",
                    Style::new().fg(GRAY),
                )))
                .gauge_style(Style::new().fg(GOLD).bg(Color::Rgb(20, 20, 20)))
                .ratio(vram.clamp(0.0, 1.0))
                .label(Span::styled(
                    format!("{:.1} / {:.1} GB", gpu.mem_used / 1024.0, gpu.mem_total / 1024.0),
                    Style::new().fg(GRAY),
                )),
            g[1],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(format!("  {}°C ", gpu.temp), Style::new().fg(if gpu.temp > 80 { RED } else { GRAY })),
                Span::styled(format!("· {:.0} W", gpu.watts), Style::new().fg(GRAY)),
            ])),
            g[2],
        );
    } else {
        f.render_widget(
            Paragraph::new(" no GPU (nvidia-smi not found) ")
                .style(Style::new().fg(DIM))
                .block(Block::bordered().border_style(DIM)),
            bot[1],
        );
    }

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " tab next run · q quit · refreshes every 0.7s · GPU every 2s",
            Style::new().fg(DIM),
        ))),
        rows[3],
    );
}
