<#
.SYNOPSIS
    AGENTIC-QUANT Runner for Windows
.DESCRIPTION
    One-click script to install dependencies and run AGENTIC-QUANT on Windows.
    Supports 3 modes: full, backend-only, dev-ui

.PARAMETER Mode
    "full"       = Docker + Backend + Frontend (default)
    "backend"    = Docker + Backend only
    "dev-ui"     = UI dev server (Tauri optional)

.PARAMETER SkipRedis
    Skip Docker/Redis check, use in-memory fallback

.PARAMETER SkipBuild
    Skip frontend build (use existing dist/)

.EXAMPLE
    .\run_windows.ps1
    .\run_windows.ps1 -Mode backend
    .\run_windows.ps1 -Mode full -SkipRedis
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
$PYTHON_MIN = "3.11"
$LOG_FILE = Join-Path $SCRIPT_DIR "logs\run.log"
$PID_FILE = Join-Path $SCRIPT_DIR "backend.pid"

# Colors
$GREEN  = "Green"
$YELLOW = "Yellow"
$RED    = "Red"
$CYAN   = "Cyan"

# =============================================================================
# Helper Functions
# =============================================================================
function Write-Step {
    param([string]$Message, [string]$Color = $CYAN)
    Write-Host "`n>> $Message" -ForegroundColor $Color
}

function Write-OK {
    Write-Host "   [OK]" -ForegroundColor $GREEN
}

function Write-Skip {
    Write-Host "   [SKIP]" -ForegroundColor $YELLOW
}

function Write-Fail {
    param([string]$Message)
    Write-Host "   [FAIL] $Message" -ForegroundColor $RED
}

function Test-Command {
    param([string]$Command)
    return (Get-Command $Command -ErrorAction SilentlyContinue) -ne $null
}

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
Write-Host @"
  ╔═══════════════════════════════════════════╗
  ║        AGENTIC-QUANT v0.1.0              ║
  ║     AI Multi-Agent Trading System         ║
  ╚═══════════════════════════════════════════╝
"@ -ForegroundColor $CYAN
Write-Host "Mode: $Mode | SkipRedis: $SkipRedis | SkipBuild: $SkipBuild`n" -ForegroundColor $YELLOW

if (!$SkipRedis) {
    Write-Host "╔═══════════════════════════════════════════╗" -ForegroundColor $GREEN
    Write-Host "║  YEU CAU: Docker Desktop phai dang chay    ║" -ForegroundColor $GREEN
    Write-Host("║  pull request: docker compose up -d") -ForegroundColor $GREEN
    Write-Host "╚═══════════════════════════════════════════╝" -ForegroundColor $GREEN
}

Start-Log

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================
Write-Step "Step 1/7: Kiem tra prerequisites..." -Color $YELLOW

$allGood = $true

# Python
if (Test-Command "python") {
    if (Test-PythonVersion) {
        Write-OK "Python $(python --version 2>&1)"
    } else {
        Write-Fail "Can Python >= $PYTHON_MIN"
        Write-Host "   => Tai python tu https://www.python.org/downloads/" -ForegroundColor $RED
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
    Write-Host "   => Tai tu https://nodejs.org/" -ForegroundColor $RED
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
            Write-Fail "Docker Desktop not running"
            Write-Host "   => Khoi dong Docker Desktop truoc, hoac dung -SkipRedis" -ForegroundColor $YELLOW
            $allGood = $false
        }
    } else {
        Write-Fail "Docker not installed"
        Write-Host "   => Tai tu https://www.docker.com/products/docker-desktop/" -ForegroundColor $RED
        Write-Host "   => Hoac chay voi -SkipRedis de dung in-memory fallback" -ForegroundColor $YELLOW
        $allGood = $false
    }
}

if (!$allGood -and !$SkipRedis) {
    Write-Host "`n[!] Nhan Enter de thu lai voi -SkipRedis, hoac Ctrl+C de thoat" -ForegroundColor $YELLOW
    $key = Read-Host
    if ($key -eq "") {
        $SkipRedis = $true
        Write-Host "=> Tiep tuc voi in-memory fallback`n" -ForegroundColor $YELLOW
    } else {
        exit 1
    }
}

# =============================================================================
# Step 2: Setup Redis + Qdrant (Docker)
# =============================================================================
if (!$SkipRedis) {
    Write-Step "Step 2/7: Khoi dong Redis + Qdrant (Docker)..." -Color $YELLOW

    Set-Location $SCRIPT_DIR

    # Check if containers already running
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
            Write-Host "   $composeResult" -ForegroundColor $RED
            Write-Host "   => Tiep tuc voi in-memory fallback..." -ForegroundColor $YELLOW
            $SkipRedis = $true
        }
    }
} else {
    Write-Step "Step 2/7: Bo qua Docker (in-memory fallback)" -Color $GREEN
}

# =============================================================================
# Step 3: Python Virtual Environment
# =============================================================================
Write-Step "Step 3/7: Thiet lap Python venv..." -Color $YELLOW

if (!(Test-Path $VENV_DIR)) {
    Write-Host "   Tao venv moi..."
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Khong the tao venv"
        exit 1
    }
    Write-OK "Venv created"
} else {
    Write-OK "Venv exists"
}

# Activate venv
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
. $activateScript

# =============================================================================
# Step 4: Install Python Dependencies
# =============================================================================
Write-Step "Step 4/7: Cai dat Python dependencies..." -Color $YELLOW

$pip = Join-Path $VENV_DIR "Scripts\pip.exe"

# Core deps (always needed)
$coreDeps = @(
    "numpy>=1.26.0", "pandas>=2.2.0", "pyyaml>=6.0", "pydantic>=2.9.0",
    "loguru>=0.9.0", "msgpack>=1.0.0",
    "pyzmq>=26.0.0", "websockets>=13.0.0", "aiohttp>=3.10.0",
    "redis>=5.2.0", "aiosqlite>=0.20.0",
    "xgboost>=2.1.0", "scikit-learn>=1.5.0",
    "orjson>=3.10.0", "tenacity>=9.0.0",
    "prometheus-client>=0.21.0"
)

Write-Host "   Cai core packages..."
foreach ($dep in $coreDeps) {
    & $pip install $dep --quiet 2>&1 | Out-Null
}
if ($LASTEXITCODE -eq 0) {
    Write-OK "Core packages installed"
} else {
    Write-Fail "pip install failed"
    exit 1
}

# ML deps (optional - skip if torch fails)
Write-Host "   Cai ML packages (torch, optuna)..."
& $pip install "torch>=2.5.0" --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "PyTorch installed"
} else {
    Write-Skip "PyTorch (khong co GPU -> skip, van chay duoc)"
}

& $pip install "optuna>=4.0.0" --quiet 2>&1 | Out-Null

# Vector DB (optional)
Write-Host "   Cai Vector DB packages..."
& $pip install "qdrant-client>=1.12.0" "chromadb>=0.4.0" --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "Vector DB installed"
} else {
    Write-Skip "Vector DB (co the cai sau)"
}

# Run SQLite migrations
Write-Host "   Chay SQLite migrations..."
python -m scripts.migrations 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-OK "DB migrations done"
} else {
    Write-Skip "Migrations (DB chua tao?)"
}

# =============================================================================
# Step 5: Install Frontend Dependencies
# =============================================================================
if ($Mode -eq "full" -or $Mode -eq "dev-ui") {
    Write-Step "Step 5/7: Cai dat Frontend dependencies..." -Color $YELLOW

    Set-Location $UI_DIR

    if (!(Test-Path "node_modules\.package-lock.json")) {
        Write-Host "   npm install..."
        npm install --silent 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-OK "npm packages installed"
        } else {
            Write-Fail "npm install failed"
            Write-Host "   => Thu: cd ui && npm install" -ForegroundColor $RED
        }
    } else {
        Write-OK "node_modules exists"
    }

    # Build frontend if needed
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
            Write-OK "dist/ exists (dung -SkipBuild de bo qua)"
        }
    }

    Set-Location $SCRIPT_DIR
} else {
    Write-Step "Step 5/7: Bo qua Frontend (backend-only mode)" -Color $GREEN
}

# =============================================================================
# Step 6: Start Backend
# =============================================================================
Write-Step "Step 6/7: Khoi dong Backend Python..." -Color $YELLOW

$env:AQ_SKIP_REDIS = if ($SkipRedis) { "1" } else { "0" }

$pythonExe = Join-Path $VENV_DIR "Scripts\python.exe"

# Kill old backend if running
$oldPid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
if ($oldPid) {
    $oldProcess = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($oldProcess) {
        Write-Host "   Tat backend cu (PID $oldPid)..."
        Stop-Process -Id $oldPid -Force
    }
}

# Start backend in background
Write-Host "   Starting: python -m core.main"
$backendJob = Start-Job -ScriptBlock {
    param($exe, $dir)
    Set-Location $dir
    $env:AQ_SKIP_REDIS = if ($env:AQ_SKIP_REDIS -eq "1") { "1" } else { "0" }
    & $exe -m core.main
} -ArgumentList $pythonExe, $SCRIPT_DIR

# Wait for READY signal
$timeout = 30
$elapsed = 0
$ready = $false
while ($elapsed -lt $timeout) {
    $output = Receive-Job -Job $backendJob 2>&1
    if ($output -match "AGENTIQ_BACKEND_READY") {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 1
    $elapsed++
}

if ($ready) {
    $backendPid = $backendJob.Id
    $backendPid | Out-File -FilePath $PID_FILE -Force
    Write-OK "Backend ready (PID: $backendPid)"
    Write-Host "   WebSocket: ws://localhost:47290" -ForegroundColor $GREEN
} else {
    Write-Fail "Backend khong the ready trong ${timeout}s"
    Write-Host "   Kiem tra log: $LOG_FILE" -ForegroundColor $RED
    Write-Host "   Thu chay thu cong: python -m core.main" -ForegroundColor $YELLOW
}

# =============================================================================
# Step 7: Start Frontend / Full App
# =============================================================================
if ($Mode -eq "full") {
    Write-Step "Step 7/7: Khoi dong Tauri Desktop..." -Color $YELLOW

    $tauriCli = (Get-Command "cargo" -ErrorAction SilentlyContinue)
    if ($tauriCli) {
        Write-Host "   cargo tauri dev..."
        Set-Location (Join-Path $SCRIPT_DIR "desktop")
        
        $tauriJob = Start-Job -ScriptBlock {
            param($dir)
            Set-Location $dir
            cargo tauri dev 2>&1
        } -ArgumentList (Join-Path $SCRIPT_DIR "desktop")
        
        Write-OK "Tauri desktop starting..."
        Write-Host "   Windows: 1600x900, title: AGENTIC-QUANT" -ForegroundColor $CYAN
    } else {
        Write-Fail "Rust/Cargo not found"
        Write-Host "   => Cai tu https://rustup.rs/" -ForegroundColor $RED
        Write-Host "   => Hoac dung Mode dev-ui de chay Frontend trong browser" -ForegroundColor $YELLOW
    }
    
    Set-Location $SCRIPT_DIR

} elseif ($Mode -eq "dev-ui") {
    Write-Step "Step 7/7: Khoi dong UI Dev Server..." -Color $YELLOW

    $uiJob = Start-Job -ScriptBlock {
        param($dir)
        Set-Location $dir
        npm run dev 2>&1
    } -ArgumentList $UI_DIR

    Write-OK "Vite dev server starting..."
    Write-Host "   Mo http://localhost:3000 trong trinh duyet" -ForegroundColor $GREEN
}

# =============================================================================
# Summary
# =============================================================================
Write-Host @"

  ╔═══════════════════════════════════════════╗
  ║         AGENTIC-QUANT IS RUNNING          ║
  ╠═══════════════════════════════════════════╣
  ║                                           ║
  ║  Backend   : ws://localhost:47290         ║
"@ -ForegroundColor $GREEN

if ($Mode -eq "full") {
    Write-Host @"
  ║  Desktop   : AGENTIC-QUANT window         ║
"@ -ForegroundColor $GREEN
} elseif ($Mode -eq "dev-ui") {
    Write-Host @"
  ║  UI Dev    : http://localhost:3000        ║
"@ -ForegroundColor $GREEN
} else {
    Write-Host @"
  ║  Mode      : backend-only                 ║
"@ -ForegroundColor $GREEN
}

Write-Host @"
  ║  Redis     : localhost:6379               ║
  ║  ZMQ       : tcp://localhost:5556          ║
  ║                                           ║
  ╠═══════════════════════════════════════════╣
  ║                                           ║
  ║  Ctrl+C de dung tat ca                     ║
  ║                                           ║
  ╚═══════════════════════════════════════════╝

"@ -ForegroundColor $CYAN

# =============================================================================
# Cleanup on Ctrl+C
# =============================================================================
try {
    # Keep script running
    while ($true) {
        Start-Sleep -Seconds 10
        
        # Check backend health
        try {
            $health = Invoke-RestMethod -Uri "http://localhost:47290/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            Write-Host "   [HEALTH] Backend OK" -ForegroundColor $GREEN
        } catch {
            Write-Host "   [HEALTH] Backend check failed" -ForegroundColor $RED
        }
    }
} finally {
    Write-Host "`nDang dung AGENTIC-QUANT..." -ForegroundColor $YELLOW
    
    # Kill backend
    if ($backendJob) {
        Stop-Job $backendJob -ErrorAction SilentlyContinue
        Remove-Job $backendJob -ErrorAction SilentlyContinue
    }
    
    # Kill Tauri
    if ($tauriJob) {
        Stop-Job $tauriJob -ErrorAction SilentlyContinue
        Remove-Job $tauriJob -ErrorAction SilentlyContinue
    }
    
    # Kill UI
    if ($uiJob) {
        Stop-Job $uiJob -ErrorAction SilentlyContinue
        Remove-Job $uiJob -ErrorAction SilentlyContinue
    }
    
    Write-OK "Da dung. Hen gap lai!"
}
