// =============================================================================
// AGENTIC-QUANT — Tauri Library Entry Point
// =============================================================================

mod ipc_bridge;
mod system_tray;

use ipc_bridge::BackendStateHandle;
use std::sync::{Arc, Mutex};
use tauri::Manager;

pub use ipc_bridge::*;
pub use system_tray::*;

/// Khoi tao va chay Tauri app
pub fn run() {
    let backend_state: BackendStateHandle = Arc::new(Mutex::new(
        ipc_bridge::BackendState::default(),
    ));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(backend_state)
        .invoke_handler(tauri::generate_handler![
            ipc_bridge::start_backend,
            ipc_bridge::stop_backend,
            ipc_bridge::get_backend_status,
            ipc_bridge::check_backend_health,
            ipc_bridge::restart_backend,
            system_tray::update_tray_status,
        ])
        .setup(|app| {
            // Setup system tray
            if let Err(e) = system_tray::setup_tray(app.handle()) {
                log::warn!("Khong the setup tray: {}", e);
            }

            // Optional: auto-start backend on launch
            #[cfg(debug_assertions)]
            {
                let handle = app.handle().clone();
                let state = app.state::<BackendStateHandle>();
                if let Ok(mut state) = state.lock() {
                    // Khong auto-start trong dev mode, de user kiem soat
                    log::info!("AGENTIC-QUANT Desktop ready. Use tray menu de khoi dong backend.");
                }
            }

            log::info!("AGENTIC-QUANT Desktop khoi dong thanh cong");
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Hide instead of close on close button
                #[cfg(not(debug_assertions))]
                {
                    window.hide().ok();
                    api.prevent_close();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("Loi khoi dong Tauri app");
}
