<#
.SYNOPSIS
    AGENTIC-QUANT Runner for Windows
.DESCRIPTION
    One-click script to install dependencies and run AGENTIC-QUANT on Windows.
.PARAMETER Mode
    "full" = Docker + Backend + Frontend (default)
    "backend" = Docker + Backend only
    "dev-ui" = UI dev server (Vite)
.PARAMETER SkipRedis
    Skip Docker/Redis check, use in-memory fallback
.PARAMETER SkipBuild
    Skip frontend build (use existing dist/)
#>

param(
    [ValidateSet("full", "backend", "dev-ui")]
    [string]$Mode = "full",
    [switch]$SkipRedis = $false,
    [switch]$SkipBuild = $false
)

# =============================================================================
# Configuration
# =============================================================================
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$BACKEND_DIR = Join-Path $SCRIPT_DIR "core"
$UI_DIR = Join-Path $SCRIPT_DIR "ui"
$VENV_DIR = Join-Path $SCRIPT_DIR ".venv"
$LOG_FILE = Join-Path $SCRIPT_DIR "logs\run.log"
$PID_FILE = Join-Path $SCRIPT_DIR "backend.pid"
$PYTHON_MIN = "3.11"

# =============================================================================
# Helper Functions
# =============================================================================
function Write-Step { param([string]$Msg) Write-Host "`n>> $Msg" -ForegroundColor Cyan }
function Write-OK { Write-Host "   [OK]" -ForegroundColor Green }
function Write-Skip { Write-Host "   [SKIP]" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg) Write-Host "   [FAIL] $Msg" -ForegroundColor Red }

function Test-Command { param([string]$Cmd) return (Get-Command $Cmd -ErrorAction SilentlyContinue) -ne $null }

function Test-PythonVersion {
    try {
        $v = python --version 2>&1
        if ($v -match "(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            return ($major -eq 3 -and $minor -ge 11)
        }
    } catch {}
    return $false
}

function Start-Log {
    $logDir = Split-Path $LOG_FILE -Parent
    if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
    Start-Transcript -Path $LOG_FILE -Append | Out-Null
}

# =============================================================================
# Banner
# =============================================================================
Clear-Host
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       AGENTIC-QUANT v0.1.0" -ForegroundColor Cyan
Write-Host "    AI Multi-Agent Trading System" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Mode: $Mode | SkipRedis: $SkipRedis | SkipBuild: $SkipBuild`n" -ForegroundColor Yellow

if (!$SkipRedis) {
    Write-Host "REQUIRED: Docker Desktop must be running" -ForegroundColor Green
    Write-Host "Pull request: docker compose up -d" -ForegroundColor Green
}

Start-Log

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================
Write-Step "Step 1/7: Checking prerequisites..."

$allGood = $true

# Python
if (Test-Command "python") {
    if (Test-PythonVersion) {
        Write-OK "Python $(python --version 2>&1)"
    } else {
        Write-Fail "Need Python >= $PYTHON_MIN"
        Write-Host "   Download: https://www.python.org/downloads/" -ForegroundColor Red
        $allGood = $false
    }
} else {
    Write-Fail "Python not found"
    $allGood = $false
}

# Node
if (Test-Command "node") {
    Write-OK "Node $(node --version 2>&1)"
} else {
    Write-Fail "Node.js not found"
    $allGood = $false
}

# npm
if (Test-Command "npm") {
    Write-OK "npm $(npm --version 2>&1)"
} else {
    Write-Fail "npm not found"
    $allGood = $false
}

# git
if (Test-Command "git") {
    Write-OK "git $(git --version 2>&1)"
} else {
    Write-Skip
}

# Docker (optional)
if (!$SkipRedis) {
    if (Test-Command "docker") {
        $dockerRunning = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Docker is running"
        } else {
            Write-Fail "Docker Desktop is not running"
            Write-Host "   Start Docker Desktop or use -SkipRedis" -ForegroundColor Yellow
            $allGood = $false
        }
    } else {
        Write-Fail "Docker not installed"
        Write-Host "   Download: https://www.docker.com/products/docker-desktop/" -ForegroundColor Red
        Write-Host "   Or use -SkipRedis for in-memory fallback" -ForegroundColor Yellow
        $allGood = $false
    }
}

if (!$allGood -and !$SkipRedis) {
    Write-Host "`nPress Enter to retry with -SkipRedis, or Ctrl+C to exit" -ForegroundColor Yellow
    $key = Read-Host
    if ($key -eq "") {
        $SkipRedis = $true
        Write-Host "Continuing with in-memory fallback`n" -ForegroundColor Yellow
    } else {
        exit 1
    }
}

# =============================================================================
# Step 2: Setup Redis + Qdrant (Docker)
# =============================================================================
if (!$SkipRedis) {
    Write-Step "Step 2/7: Starting Redis + Qdrant (Docker)..."
    Set-Location $SCRIPT_DIR

    $redisRunning = docker ps --filter "name=agentic-quant-redis" --format "{{.Names}}" 2>$null
    $qdrantRunning = docker ps --filter "name=agentic-quant-qdrant" --format "{{.Names}}" 2>$null

    if ($redisRunning -and $qdrantRunning) {
        Write-OK "Containers already running"
    } else {
        Write-Host "   docker compose up -d..."
        $composeResult = docker compose up -d 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-OK "Redis :6379 + Qdrant :6333"
        } else {
            Write-Fail "Docker compose failed"
            Write-Host "   $composeResult" -ForegroundColor Red
            Write-Host "   Continuing with in-memory fallback..." -ForegroundColor Yellow
            $SkipRedis = $true
        }
    }
} else {
    Write-Step "Step 2/7: Skipping Docker (in-memory fallback)"
}

# =============================================================================
# Step 3: Python Virtual Environment
# =============================================================================
Write-Step "Step 3/7: Setting up Python venv..."

if (!(Test-Path $VENV_DIR)) {
    Write-Host "   Creating venv..."
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Cannot create venv"
        exit 1
    }
    Write-OK "Venv created at $VENV_DIR"
} else {
    Write-OK "Venv exists"
}

# Activate venv
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
. $activateScript

# =============================================================================
# Step 4: Install Python Dependencies
# =============================================================================
Write-Step "Step 4/7: Installing Python dependencies..."

$pip = Join-Path $VENV_DIR "Scripts\pip.exe"

$coreDeps = @(
    "numpy>=1.26.0", "pandas>=2.2.0", "pyyaml>=6.0", "pydantic>=2.9.0",
    "loguru>=0.9.0", "msgpack>=1.0.0",
    "pyzmq>=26.0.0", "websockets>=13.0.0", "aiohttp>=3.10.0",
    "redis>=5.2.0", "aiosqlite>=0.20.0",
    "xgboost>=2.1.0", "scikit-learn>=1.5.0",
    "orjson>=3.10.0", "tenacity>=9.0.0",
    "prometheus-client>=0.21.0"
)

Write-Host "   Installing core packages..."
foreach ($dep in $coreDeps) {
    & $pip install $dep --quiet 2>&1 | Out-Null
}
if ($LASTEXITCODE -eq 0) {
    Write-OK "Core packages installed"
} else {
    Write-Fail "pip install failed"
    exit 1
}

# ML deps (optional - torch)
Write-Host "   Installing ML packages (torch)..."
& $pip install "torch>=2.5.0" --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "PyTorch installed"
} else {
    Write-Skip "PyTorch (no GPU, can still run)"
}

& $pip install "optuna>=4.0.0" --quiet 2>&1 | Out-Null

# Vector DB (optional)
Write-Host "   Installing Vector DB packages..."
& $pip install "qdrant-client>=1.12.0" "chromadb>=0.4.0" --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "Vector DB installed"
} else {
    Write-Skip "Vector DB (can install later)"
}

# Run SQLite migrations
Write-Host "   Running SQLite migrations..."
python -m scripts.migrations 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "DB migrations done"
} else {
    Write-Skip "Migrations (DB may not exist yet)"
}

# =============================================================================
# Step 5: Install Frontend Dependencies
# =============================================================================
if ($Mode -eq "full" -or $Mode -eq "dev-ui") {
    Write-Step "Step 5/7: Installing Frontend dependencies..."
    Set-Location $UI_DIR

    if (!(Test-Path "node_modules\.package-lock.json")) {
        Write-Host "   npm install..."
        npm install --silent 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-OK "npm packages installed"
        } else {
            Write-Fail "npm install failed"
            Write-Host "   Try: cd ui && npm install" -ForegroundColor Red
        }
    } else {
        Write-OK "node_modules exists"
    }

    if ($Mode -eq "full" -and !$SkipBuild) {
        Write-Host "   npm run build..."
        if (!(Test-Path "dist\index.html")) {
            npm run build 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-OK "Frontend built (ui/dist/)"
            } else {
                Write-Fail "Frontend build failed"
            }
        } else {
            Write-OK "dist/ exists (use -SkipBuild to skip)"
        }
    }
    Set-Location $SCRIPT_DIR
} else {
    Write-Step "Step 5/7: Skipping Frontend (backend-only mode)"
}

# =============================================================================
# Step 6: Start Backend
# =============================================================================
Write-Step "Step 6/7: Starting Backend..."

$env:AQ_SKIP_REDIS = if ($SkipRedis) { "1" } else { "0" }
$pythonExe = Join-Path $VENV_DIR "Scripts\python.exe"

# Kill old backend if running
$oldPid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
if ($oldPid) {
    $oldProcess = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($oldProcess) {
        Write-Host "   Killing old backend (PID $oldPid)..."
        Stop-Process -Id $oldPid -Force
    }
}

# Start backend process
Write-Host "   Starting: python -m core.main"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $pythonExe
$psi.Arguments = "-m core.main"
$psi.WorkingDirectory = $SCRIPT_DIR
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.EnvironmentVariables["AQ_SKIP_REDIS"] = $env:AQ_SKIP_REDIS

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$proc.Start() | Out-Null

$backendPid = $proc.Id
$backendPid | Out-File -FilePath $PID_FILE -Force

# Wait for AGENTIQ_BACKEND_READY
$timeout = 30
$ready = $false
$output = ""
for ($i = 0; $i -lt $timeout; $i++) {
    if ($proc.HasExited) { break }
    $line = $proc.StandardOutput.ReadLine()
    if ($line -match "AGENTIQ_BACKEND_READY") {
        $ready = $true
        break
    }
    if ($line) { $output += $line + "`n" }
    Start-Sleep -Seconds 1
}

if ($ready) {
    Write-OK "Backend ready (PID: $backendPid)"
    Write-Host "   WebSocket: ws://localhost:47290" -ForegroundColor Green
    Write-Host "   ZMQ Pull: tcp://localhost:5556" -ForegroundColor Green
} else {
    Write-Fail "Backend did not become ready in ${timeout}s"
    Write-Host "   Check log: $LOG_FILE" -ForegroundColor Red
    Write-Host "   Try manual: python -m core.main" -ForegroundColor Yellow
}

# =============================================================================
# Step 7: Start Frontend / Full App
# =============================================================================
if ($Mode -eq "full") {
    Write-Step "Step 7/7: Starting Tauri Desktop..."
    if (Test-Command "cargo") {
        Write-Host "   cargo tauri dev..."
        Set-Location (Join-Path $SCRIPT_DIR "desktop")
        $tauriProc = Start-Process -NoNewWindow -FilePath "cargo" -ArgumentList "tauri dev" -PassThru
        Write-OK "Tauri desktop starting..."
        Write-Host "   Window: 1600x900, title: AGENTIC-QUANT" -ForegroundColor Cyan
    } else {
        Write-Fail "Rust/Cargo not found"
        Write-Host "   Install from https://rustup.rs/" -ForegroundColor Red
        Write-Host "   Or use mode 'dev-ui' for browser" -ForegroundColor Yellow
    }
    Set-Location $SCRIPT_DIR

} elseif ($Mode -eq "dev-ui") {
    Write-Step "Step 7/7: Starting UI Dev Server..."
    $uiJob = Start-Job -ScriptBlock {
        param($dir) Set-Location $dir; npm run dev 2>&1
    } -ArgumentList $UI_DIR
    Write-OK "Vite dev server starting..."
    Write-Host "   Open http://localhost:3000 in browser" -ForegroundColor Green
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n================================================" -ForegroundColor Green
Write-Host "       AGENTIC-QUANT IS RUNNING" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Backend : ws://localhost:47290" -ForegroundColor Green

if ($Mode -eq "full") {
    Write-Host "  Desktop : AGENTIC-QUANT window" -ForegroundColor Green
} elseif ($Mode -eq "dev-ui") {
    Write-Host "  UI Dev  : http://localhost:3000" -ForegroundColor Green
} else {
    Write-Host "  Mode    : backend-only" -ForegroundColor Green
}

Write-Host "  Redis   : localhost:6379" -ForegroundColor Green
Write-Host "  ZMQ     : tcp://localhost:5556" -ForegroundColor Green
Write-Host "-----------------------------------------------" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop everything" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Green

# =============================================================================
# Keep running + health check
# =============================================================================
try {
    while ($true) {
        Start-Sleep -Seconds 10
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:47290/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            Write-Host "   [HEALTH] Backend OK" -ForegroundColor Green
        } catch {
            Write-Host "   [HEALTH] Backend check failed" -ForegroundColor Red
        }
    }
} finally {
    Write-Host "`nShutting down AGENTIC-QUANT..." -ForegroundColor Yellow
    if ($proc -and !$proc.HasExited) { $proc.Kill() }
    Write-OK "Done. See you later!"
}
