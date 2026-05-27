// =============================================================================
// AGENTIC-QUANT — System Tray
// Hien thi tray icon voi cac trang thai:
//   - Green:  Normal
//   - Yellow: Warning / active_guardrail
//   - Red:    Critical / MODEL_DEGRADED / feed_failure
// Menu context:
//   - Show/Hide cua so
//   - Restart Backend
//   - Quit
// Notification khi co High Impact news trong 15 phut
// =============================================================================

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager,
};

pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let show_item = MenuItem::with_id(app, "show", "Show / Hide", true, None::<&str>)?;
    let restart_item = MenuItem::with_id(app, "restart", "Restart Backend", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&show_item, &restart_item, &quit_item])?;

    let _tray = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("AGENTIC-QUANT")
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_visible().unwrap_or(false) {
                        window.hide().ok();
                    } else {
                        window.show().ok();
                        window.set_focus().ok();
                    }
                }
            }
            "restart" => {
                let _ = app.emit("tray-restart-backend", ());
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_visible().unwrap_or(false) {
                        window.hide().ok();
                    } else {
                        window.show().ok();
                        window.set_focus().ok();
                    }
                }
            }
        })
        .build(app)?;

    Ok(())
}

/// Cap nhat tray icon theo trang thai he thong
#[tauri::command]
pub fn update_tray_status(
    app: AppHandle,
    model_degraded: bool,
    feed_failure: bool,
    active_guardrail: bool,
    is_connected: bool,
) -> Result<(), String> {
    let tooltip = if !is_connected {
        "AGENTIC-QUANT — Disconnected"
    } else if feed_failure {
        "AGENTIC-QUANT — Feed Failure"
    } else if model_degraded {
        "AGENTIC-QUANT — Model Degraded"
    } else if active_guardrail {
        "AGENTIC-QUANT — News Guardrail Active"
    } else {
        "AGENTIC-QUANT — Running"
    };

    if let Some(tray) = app.tray_by_id("main") {
        tray.set_tooltip(Some(tooltip)).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Gui notification khi co tin quan trong
#[tauri::command]
pub fn send_news_notification(app: &AppHandle, event_name: String, minutes_until: i32) {
    if let Some(tray) = app.tray_by_id("main") {
        let title = "AGENTIC-QUANT — News Alert";
        let body = format!("{} trong {} phut", event_name, minutes_until);
        tray.notify(Some(&tauri::tray::Notification::new(
            app.config(),
            title,
            &body,
        )))
        .ok();
    }
}
