# How to Use Nexa ToolKit

Nexa ToolKit can be operated in two modes: a user-friendly interactive Terminal User Interface (TUI) or via direct Command-Line Interface (CLI) commands for scripting and automation.

## Prerequisites

1.  Ensure you have Python 3.8+ installed.
2.  Install the required Python packages:
    ```bash
    pip install torch safetensors fastapi uvicorn
    ```
3.  Build the Go application:
    ```bash
    go build -o nexa-toolkit.exe main.go
    ```

## Interactive TUI Mode

The TUI mode provides a guided experience for all operations.

### Starting the TUI

*   **To select a file within the TUI:**
    Run the executable without any arguments. It will first prompt you for a model file path.
    ```bash
    ./nexa-toolkit.exe
    ```

*   **To start with a file preloaded:**
    Provide the path to your model file as an argument.
    ```bash
    ./nexa-toolkit.exe C:\path\to\your\model.safetensors
    ```

### Using the TUI

1.  Use the **arrow keys** (↑/↓) to navigate the list of operations.
2.  Press **Enter** to select an operation and execute it.
3.  Press **'q'** or **Ctrl+C** to quit at any time.
4.  From a results or error screen, press **Enter** or **'b'** to go back to the main menu.

## Command-Line (CLI) Mode

The CLI mode is ideal for direct execution and scripting. The Python backend server is started and stopped automatically for each command.

### General Syntax

```bash
./nexa-toolkit.exe [path/to/model.pt] <command> [flags]
```

### Commands & Examples

#### 1. Analyze

Inspect a model's metadata, including architecture, size, and tensor details.

```bash
./nexa-toolkit.exe model.pt analyze
```

#### 2. Convert

Convert a model between `.pt`/`.pth` and `.safetensors` formats.

*   **Basic Conversion (to .safetensors by default):**
    ```bash
    ./nexa-toolkit.exe model.pt convert
    ```
    This will create `model.safetensors`.

*   **Specify Output Format and Path:**
    ```bash
    ./nexa-toolkit.exe model.safetensors convert --format pytorch --output new_model.pt
    ```

#### 3. Quantize

Reduce model size by changing its precision.

*   **Default Quantization (to fp16):**
    ```bash
    ./nexa-toolkit.exe model.pt quantize
    ```
    This creates `model.fp16.pt`.

*   **Specify Method and Output:**
    ```bash
    ./nexa-toolkit.exe model.pt quantize --method bf16 --output model_bf16.pt
    ```
    Supported methods: `fp16`, `bf16`, `int8`.

#### 4. Prune

Reduce model parameters by removing low-magnitude weights. (Note: This is a basic implementation).

```bash
./nexa-toolkit.exe model.pt prune
```

#### 5. Check Environment

Verify that Python and all required dependencies are installed correctly. This command does not require a model file.

```bash
./nexa-toolkit.exe check
```
