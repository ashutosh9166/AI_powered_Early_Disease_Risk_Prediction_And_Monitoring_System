#!/usr/bin/env bash
# ==============================================================================
# VitalGuard AI — Remote Patient Monitoring System Launcher
# Starts FastAPI Backend (port 8000) and Astro Frontend UI (port 4321)
# ==============================================================================

set -e

# Change directory to project root
cd "$(dirname "$0")"

# 1. Detect Python virtual environment or system Python
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "❌ Error: Python not found! Please create a .venv or ensure Python is in your PATH."
    exit 1
fi

# 2. Check for npm
if ! command -v npm >/dev/null 2>&1; then
    echo "❌ Error: npm is not installed or not found in PATH."
    exit 1
fi

echo "======================================================================"
echo "  VitalGuard AI — Remote Patient Monitoring System"
echo "======================================================================"
echo "  Detected Python : $PYTHON"
echo "  Starting services..."
echo "======================================================================"
echo ""

# Helper to open browser URL cross-platform
open_browser() {
    local url="http://localhost:4321"
    if command -v start >/dev/null 2>&1; then
        start "$url" 2>/dev/null || true
    elif command -v explorer.exe >/dev/null 2>&1; then
        explorer.exe "$url" 2>/dev/null || true
    elif command -v open >/dev/null 2>&1; then
        open "$url" 2>/dev/null || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" 2>/dev/null || true
    fi
}

# 3. Launch FastAPI Backend
echo " [1/2] 🚀 Starting FastAPI Backend API on http://127.0.0.1:8000..."
"$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# 4. Launch Astro Frontend
echo " [2/2] 🚀 Starting Astro Frontend UI on http://127.0.0.1:4321..."
(cd frontend && npm run dev -- --host 127.0.0.1 --port 4321) &
FRONTEND_PID=$!

# Optional: Launch continuous telemetry simulator if --sim or --simulator passed
SIM_PID=""
if [ "$1" = "--sim" ] || [ "$1" = "--simulator" ]; then
    echo " [3/3] 📡 Starting Wearable Telemetry Simulator..."
    "$PYTHON" simulator/data_simulator.py &
    SIM_PID=$!
fi

# Automatically open the browser after a short delay
(sleep 3 && open_browser) &
BROWSER_LAUNCH_PID=$!

# Clean shutdown function on Ctrl+C
cleanup() {
    echo ""
    echo "======================================================================"
    echo "  Shutting down VitalGuard AI services..."
    echo "======================================================================"
    
    [ -n "$SIM_PID" ] && kill "$SIM_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null || true
    
    # On Windows, kill any orphaned uvicorn/node child processes if needed
    if command -v taskkill >/dev/null 2>&1; then
        taskkill //F //T //PID "$BACKEND_PID" 2>/dev/null || true
        taskkill //F //T //PID "$FRONTEND_PID" 2>/dev/null || true
    fi
    
    echo "  All services stopped cleanly. Bye!"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo ""
echo "======================================================================"
echo "  VitalGuard AI System is running!"
echo "======================================================================"
echo "  🌐 Web Portal:         http://localhost:4321"
echo "  🩺 Doctor Dashboard:   http://localhost:4321/doctor/dashboard"
echo "  📱 Patient App:        http://localhost:4321/patient/home"
echo "  ⚙️  Backend API Docs:   http://localhost:8000/docs"
echo ""
echo "  🔑 Demo Credentials:"
echo "     Doctor:   doctor@demo.com  / password123"
echo "     Patient:  patient@demo.com / password123"
echo "======================================================================"
echo "  (Opening your browser automatically at http://localhost:4321...)"
echo "  Press Ctrl+C anytime to stop all services."
echo "======================================================================"
echo ""

# Wait for background processes
wait
