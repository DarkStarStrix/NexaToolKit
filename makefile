# Makefile for the Nexa Toolkit project

# Define the output binary name
BINARY_NAME=nexa-toolkit.exe
PYTHON=python3

# Default target
all: build

# Build the Go application
build:
	@echo "Building Go application..."
	go build -o $(BINARY_NAME) .

# Run the application (assumes backend dependencies are installed)
run: build
	@echo "Starting Nexa Toolkit..."
	./$(BINARY_NAME)

# Install all dependencies (Go and Python)
install:
	@echo "Installing Go dependencies..."
	go mod tidy
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install -r requirements.txt

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	go clean
	rm -f $(BINARY_NAME)

# A simple command to check the environment
check:
	@echo "Checking environment..."
	go version
	$(PYTHON) --version

.PHONY: all build run install clean check
