#!/bin/bash

# Nexa ToolKit Launcher
# This script starts the backend services and launches the TUI.

set -e

# --- Helper Functions ---
echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}

# --- Main Script ---
echo_green "🚀 Starting Nexa ToolKit..."

# 1. Start backend services with Docker Compose
echo_green "\n[1/2] Starting backend services..."
docker-compose up -d

# Give the services a moment to initialize
sleep 5

# 2. Build and run the Go frontend
echo_green "\n[2/2] Building and launching the frontend TUI..."
go build -o nexa-toolkit.exe Main.go

# Pass any arguments from this script to the executable
./nexa-toolkit.exe "$@"

# 3. Clean up when the TUI exits
echo_green "\nShutting down backend services..."
docker-compose down

echo_green "Goodbye!"
