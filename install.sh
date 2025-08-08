#!/bin/bash
# Nexa Toolkit Installation Script

echo "🚀 Installing Nexa Toolkit..."

# Check prerequisites
echo "Checking prerequisites..."

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check Go installation
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.18+ first."
    exit 1
else
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    echo "✅ Go $GO_VERSION found"
fi

# Clone the repository if it doesn't exist
if [ ! -d ".git" ]; then
    echo "📥 Cloning the repository..."
    git clone https://github.com/your-username/nexa-toolkit.git .
else
    echo "✅ Repository already exists"
fi

# Set up the environment file
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping creation."
else
    cp .env.example .env
    echo "Created .env file from .env.example."
fi

echo "Please enter your license key (this will be stored in your .env file):"
read -r license_key
# Use sed to update the NEXA_LICENSE_KEY in the .env file
sed -i'' -e "s/^NEXA_LICENSE_KEY=.*/NEXA_LICENSE_KEY=$license_key/" .env
echo "License key saved."

# Build Docker containers
echo "Building Docker containers..."
docker-compose build

# Install Python dependencies
echo "Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "❌ pip is not available. Please install pip first."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Install Go dependencies and build
echo "Installing Go dependencies and building..."
go mod tidy
if [ $? -ne 0 ]; then
    echo "❌ Failed to download Go dependencies"
    exit 1
fi

go build -o nexa-toolkit Main.go
if [ $? -eq 0 ]; then
    echo "✅ Nexa Toolkit built successfully"
else
    echo "❌ Failed to build Nexa Toolkit"
    exit 1
fi

# Make backend script executable
chmod +x Backend.py

# Test installation
echo "Testing installation..."
./nexa-toolkit check
if [ $? -eq 0 ]; then
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Usage:"
    echo "  Interactive mode: ./nexa-toolkit [model_file]"
    echo "  Direct CLI:      ./nexa-toolkit [model_file] <operation> [flags]"
    echo ""
    echo "Example:"
    echo "  ./nexa-toolkit model.pt analyze"
    echo "  ./nexa-toolkit model.pt convert --format safetensors"
    echo "  ./nexa-toolkit model.pt quantize --method fp16"
else
    echo "❌ Installation test failed"
    exit 1
fi
