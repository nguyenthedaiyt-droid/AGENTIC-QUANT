// =============================================================================
// AGENTIC-QUANT — Tauri IPC Bridge
// Giao tiep giua Tauri (Rust) va Python backend thong qua:
//   - Spawn Python subprocess
//   - Doc stdout de phat hien "AGENTIQ_BACKEND_READY"
//   - Truyen WebSocket port sang React
//   - Xu ly stderr log
//   - Crash detection va auto-restart
// =============================================================================

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};

// --- State ---

pub struct BackendState {
    process: Option<Child>,
    started_at: Option<Instant>,
    ws_port: Option<u16>,
    restart_count: u32,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            process: None,
            started_at: None,
            ws_port: None,
            restart_count: 0,
        }
    }
}

type BackendStateHandle = Arc<Mutex<BackendState>>;

// --- Tien ich ---

/// Lay duong dan toi Python project (tinh tu thu muc desktop/)
fn get_python_project_path() -> PathBuf {
    let mut path = std::env::current_dir().unwrap_or_default();
    // desktop/ -> project root
    path.pop(); // desktop
    path
}

/// Lay thu muc chua app data
fn get_app_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("agentic-quant")
}

/// Lay thu muc log files
fn get_log_dir() -> PathBuf {
    get_app_data_dir().join("logs")
}

// --- Lenh khoi dong Python ---

/// Tao command khoi dong Python backend
fn build_python_command() -> Command {
    let project_path = get_python_project_path();
    let python_exe = std::env::var("PYTHON_EXE").unwrap_or_else(|_| "python".to_string());

    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", "cd", &project_path.to_string_lossy(), "&&"]);
        c.arg(&python_exe);
        c
    } else {
        let mut c = Command::new("bash");
        c.args(["-c", &format!("cd '{}' && ${{PYTHON_EXE:-python}}", project_path.to_string_lossy())]);
        c
    };

    cmd.arg("-m")
        .arg("core.main")
        .current_dir(&project_path);

    // Environment
    let mut env = std::env::vars().collect::<Vec<_>>();
    env.push(("AGENTIQ_BACKEND".to_string(), "1".to_string()));
    for (k, v) in env {
        cmd.env(&k, &v);
    }

    // Stdio
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio.piped());
    cmd.stderr(Stdio.piped());

    #[cfg(not(debug_assertions))]
    {
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped().unwrap_or_else(|_| Stdio::null()));
    }

    cmd
}

// --- WebSocket port detection ---

const READY_SIGNAL: &str = "AGENTIQ_BACKEND_READY";
const PORT_FILE: &str = "aq_ws_port.txt";

/// Doc port tu file tam
fn read_ws_port() -> Option<u16> {
    let port_file = get_app_data_dir().join(PORT_FILE);
    std::fs::read_to_string(&port_file)
        .ok()?
        .trim()
        .parse()
        .ok()
}

// --- Core functions ---

/// Khoi dong Python backend process
#[tauri::command]
pub fn start_backend(app: AppHandle, state: State<'_, BackendStateHandle>) -> Result<String, String> {
    let mut backend = state.lock().map_err(|e| e.to_string())?;

    if backend.process.is_some() {
        return Ok(format!(
            "Backend da chay (restart #{})",
            backend.restart_count
        ));
    }

    let mut cmd = build_python_command();

    let log_dir = get_log_dir();
    std::fs::create_dir_all(&log_dir).ok();

    let log_file_path = log_dir.join(format!(
        "backend_{}.log",
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    ));
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .ok();

    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "Loi khoi dong Python: {}. Hay dam bao Python da cai dat va pyproject.toml da duoc install.",
            e
        )
    })?;

    // Doc stdout de phat hien ready signal
    let stdout = child.stdout.take();
    let app_handle = app.clone();

    std::thread::spawn(move || {
        if let Some(stdout) = stdout {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                // Log vao file
                if let (Some(ref mut f), _) = (
                    log_file.as_mut(),
                    log_file_path.to_str(),
                ) {
                    let _ = writeln!(f, "{}", line);
                }

                // Kiem tra ready signal
                if line.contains(READY_SIGNAL) {
                    if let Some(port) = read_ws_port() {
                        // Chap nhan port vao frontend
                        let _ = app_handle.emit("backend-ready", port);
                    }
                }
            }
        }

        // Doc stderr
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            let mut stderr_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_dir.join("backend_stderr.log"))
                .ok();

            for line in reader.lines().map_while(Result::ok) {
                if let (Some(ref mut f), _) = (stderr_file.as_mut(), log_file_path.to_str()) {
                    let _ = writeln!(f, "[{}] {}", chrono::Local::now().format("%H:%M:%S"), line);
                }
            }
        }
    });

    backend.process = Some(child);
    backend.started_at = Some(Instant::now());
    backend.restart_count += 1;

    let started = backend.started_at
        .map(|t| format!("{:.1}s ago", t.elapsed().as_secs_f32()))
        .unwrap_or_default();

    Ok(format!(
        "Backend khoi dong (lan #{}) - {started}",
        backend.restart_count
    ))
}

/// Dung Python backend
#[tauri::command]
pub fn stop_backend(state: State<'_, BackendStateHandle>) -> Result<String, String> {
    let mut backend = state.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = backend.process.take() {
        child.start_kill().ok();
        std::thread::sleep(Duration::from_millis(500));
        child.wait().ok();
        backend.process = None;
        backend.ws_port = None;
        backend.started_at = None;
        return Ok("Backend da dung".to_string());
    }

    Ok("Backend chua chay".to_string())
}

/// Lay trang thai backend
#[tauri::command]
pub fn get_backend_status(state: State<'_, BackendStateHandle>) -> Result<serde_json::Value, String> {
    let backend = state.lock().map_err(|e| e.to_string())?;

    let is_running = backend.process.as_mut().map(|p| p.poll().is_ok()).unwrap_or(false);
    let uptime = backend.started_at.map(|t| t.elapsed().as_secs()).unwrap_or(0);
    let port = backend.ws_port.or_else(read_ws_port);

    Ok(serde_json::json!({
        "running": is_running,
        "uptime_seconds": uptime,
        "ws_port": port,
        "restart_count": backend.restart_count,
        "started_at": backend.started_at.map(|t| t.elapsed().as_secs_f32()),
    }))
}

/// Kiem tra va restart neu crashed
#[tauri::command]
pub fn check_backend_health(state: State<'_, BackendStateHandle>) -> Result<String, String> {
    let mut backend = state.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = backend.process {
        match child.poll() {
            Some(_) => {
                // Process da ket thuc
                backend.process = None;
                backend.ws_port = None;
                backend.started_at = None;
                return Ok("CRASHED".to_string());
            }
            None => return Ok("RUNNING".to_string()),
        }
    }

    Ok("NOT_RUNNING".to_string())
}

/// Restart backend
#[tauri::command]
pub fn restart_backend(
    app: AppHandle,
    state: State<'_, BackendStateHandle>,
) -> Result<String, String> {
    stop_backend(state.clone())?;
    std::thread::sleep(Duration::from_secs(1));
    start_backend(app, state)
}
