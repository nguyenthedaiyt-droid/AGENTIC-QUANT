// =============================================================================
// AGENTIC-QUANT — Tauri Desktop Entry Point
// =============================================================================

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Khoi tao logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    agentic_quant_desktop_lib::run()
}
