# ==============================================================================
# VitalGuard AI - Remote Patient Monitoring System Launcher
# Starts FastAPI Backend (port 8000) and Astro Frontend (port 4321)
# ==============================================================================

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location -Path $ProjectRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  VitalGuard AI - Remote Patient Monitoring System Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Detect Python virtual environment or system Python
$PythonExe = $null
if (Test-Path "$ProjectRoot\.venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $PythonExe = "python"
    } else {
        Write-Host "Error: Python was not found in .venv or system PATH." -ForegroundColor Red
        exit 1
    }
}

# 2. Check for npm
$npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCmd) {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
}
if (-not $npmCmd) {
    Write-Host "Error: npm was not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "  Detected Python : $PythonExe" -ForegroundColor Green
Write-Host "  Starting services..." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 3. Launch FastAPI Backend
Write-Host " [1/2] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" -PassThru

# 4. Launch Astro Frontend
Write-Host " [2/2] Starting Astro Frontend UI on http://127.0.0.1:4321..." -ForegroundColor Yellow
$frontendProcess = Start-Process -FilePath $npmCmd.Source -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", "4321" -WorkingDirectory "$ProjectRoot\frontend" -PassThru

# 5. Optional Telemetry Simulator
$simProcess = $null
if ($args -and ($args -contains "--sim" -or $args -contains "--simulator")) {
    Write-Host " [3/3] Starting Wearable Telemetry Simulator..." -ForegroundColor Yellow
    $simProcess = Start-Process -FilePath $PythonExe -ArgumentList "simulator/data_simulator.py" -PassThru
}

# 6. Open browser after startup pause
Start-Sleep -Seconds 3
Start-Process "http://localhost:4321"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  VitalGuard AI System is running!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Web Portal:         http://localhost:4321" -ForegroundColor White
Write-Host "  Doctor Dashboard:   http://localhost:4321/doctor/dashboard" -ForegroundColor White
Write-Host "  Patient App:        http://localhost:4321/patient/home" -ForegroundColor White
Write-Host "  Backend API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  Demo Credentials:" -ForegroundColor Yellow
Write-Host "     Doctor:   doctor@demo.com  / password123" -ForegroundColor Gray
Write-Host "     Patient:  patient@demo.com / password123" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Cyan

try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($backendProcess.HasExited -or $frontendProcess.HasExited) {
            break
        }
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping VitalGuard AI services..." -ForegroundColor Yellow
    if ($simProcess -and -not $simProcess.HasExited) {
        Stop-Process -Id $simProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All services stopped." -ForegroundColor Green
}
