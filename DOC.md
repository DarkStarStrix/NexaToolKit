# Nexa ToolKit - Full Project Documentation

**Version:** 1.0.0

## 1. Project Overview

Nexa ToolKit is a high-performance, cross-platform tool designed for machine learning engineers and researchers to analyze, optimize, and manage ML models. It combines a fast, native Go frontend with a powerful Python backend, offering both an intuitive Terminal User Interface (TUI) and a scriptable Command-Line Interface (CLI).

### Key Features

-   **Model Analysis:** Get detailed statistics about your models, including parameter count, memory size, architecture, and tensor information.
-   **Format Conversion:** Seamlessly convert models between PyTorch (`.pt`, `.pth`) and SafeTensors (`.safetensors`) formats.
-   **Quantization:** Reduce model size and potentially speed up inference by converting weights to lower precision formats like FP16, BF16, or INT8.
-   **Weight Pruning:** A basic implementation to reduce model parameters by zeroing out low-magnitude weights.
-   **Environment Check:** Quickly validate that all necessary dependencies are installed and available.
-   **Dual-Mode Operation:** Use the interactive TUI for guided workflows or the direct CLI for automation and scripting.

## 2. Architecture

Nexa ToolKit uses a client-server architecture that runs entirely on the local machine.

-   **Go Frontend (`main.go`):** A compiled native executable that serves as the user interface. It manages user input, displays results, and controls the lifecycle of the Python backend.
    -   **Libraries:** `Cobra` for CLI, `Bubble Tea` for TUI.
-   **Python Backend (`Backend.py`):** A lightweight FastAPI server that exposes the core model processing logic via a REST API. It leverages PyTorch and SafeTensors to perform all ML-related tasks.
    -   **Libraries:** `FastAPI`, `Uvicorn`, `PyTorch`, `SafeTensors`.
-   **Communication:** The Go frontend sends HTTP requests with JSON payloads to the Python backend on `http://127.0.0.1:8000`. The backend performs the requested operation and returns a JSON response.

*(For a more detailed diagram and explanation, see `HOW_IT_WORKS.md`)*

## 3. Installation and Setup

### Prerequisites

-   Go (1.18+)
-   Python (3.8+) and `pip`
-   Git

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Nexa-ToolKit
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file containing `torch`, `safetensors`, `fastapi`, `uvicorn`)*

3.  **Build the Go application:**
    This command compiles the frontend into a single executable file.
    -   On Windows:
        ```bash
        go build -o nexa-toolkit.exe main.go
        ```
    -   On macOS/Linux:
        ```bash
        go build -o nexa-toolkit main.go
        ```

## 4. Usage Guide

*(For a more detailed guide with examples, see `HOW_TO_USE.md`)*

### Interactive TUI Mode

Run the executable with or without a file path to enter the interactive mode. Use arrow keys to navigate and Enter to select.

```bash
# Start and then select a file
./nexa-toolkit.exe

# Start with a file pre-loaded
./nexa-toolkit.exe C:\path\to\model.pt
```

### CLI Mode

Execute commands directly for scripting.

**Syntax:** `./nexa-toolkit.exe [model_file] <command> [flags]`

**Commands:**
-   `analyze`: Inspect model metadata.
-   `convert`: Change model format.
-   `quantize`: Reduce model precision.
-   `prune`: Remove weights.
-   `check`: Verify environment dependencies.

**Flags:**
-   `--output <path>`: Specify the output file path.
-   `--format <type>`: Target format for conversion (`safetensors` or `pytorch`).
-   `--method <type>`: Quantization method (`fp16`, `bf16`, `int8`).
-   `--verbose`: Print detailed JSON results in CLI mode.

## 5. Code Structure

-   `main.go`: Contains all the Go source code for the frontend, including CLI definitions (Cobra), TUI logic (Bubble Tea), and backend communication.
-   `Backend.py`: Contains the Python source code for the backend server, including API endpoint definitions (FastAPI) and model processing logic (PyTorch).
-   `go.mod` / `go.sum`: Go module files for managing Go dependencies.
-   `requirements.txt`: (To be created) Lists Python dependencies.
