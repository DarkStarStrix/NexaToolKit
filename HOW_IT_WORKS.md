# How Nexa ToolKit Works

Nexa ToolKit employs a dual-component architecture: a frontend written in Go and a backend processing engine written in Python. This design leverages the strengths of both languages: Go for creating fast, native, and cross-platform command-line applications, and Python for its rich ecosystem of machine learning libraries.

## Architecture Overview

```
+---------------------------------+
|         User (Terminal)         |
+---------------------------------+
                |
                v
+---------------------------------+
|      Go Frontend (main.go)      |
| - TUI (Bubble Tea)              |
| - CLI (Cobra)                   |
| - Starts/Stops Python Backend   |
+---------------------------------+
                |
                v (HTTP Request: localhost:8000)
+---------------------------------+
|    Python Backend (Backend.py)  |
| - FastAPI Server               |
| - Model Processing (PyTorch)    |
| - Logic (Analyze, Convert, etc.)|
+---------------------------------+
                |
                v
+---------------------------------+
|      Model Files (.pt, .st)     |
+---------------------------------+
```

### 1. Go Frontend (`main.go`)

The frontend is the user-facing part of the application, built as a single, native executable.

-   **CLI Framework (`github.com/spf13/cobra`):** Cobra is used to create a powerful command-line interface, handling commands (`analyze`, `convert`), arguments (file paths), and flags (`--output`).
-   **TUI Framework (`github.com/charmbracelet/bubbletea`):** For interactive mode, Bubble Tea provides a simple and elegant framework for building terminal user interfaces. It manages the application state, user input, and rendering the UI components.
-   **Backend Management:** When the Go application starts (either in TUI or CLI mode), it launches the Python backend script (`Backend.py`) as a background subprocess. It waits for the backend to become available by polling its health-check endpoint. When the Go application exits, it terminates the Python subprocess.
-   **Communication:** It communicates with the backend by sending JSON-formatted HTTP requests to the FastAPI server.

### 2. Python Backend (`Backend.py`)

The backend is a lightweight web server that exposes the core model processing logic.

-   **Web Framework (`fastapi`):** FastAPI is used to create a simple, fast, and modern API. It defines endpoints like `/analyze`, `/convert`, etc., that correspond to the operations available in the toolkit. It handles request validation using Pydantic models.
-   **Web Server (`uvicorn`):** Uvicorn is the ASGI server that runs the FastAPI application, handling the HTTP traffic.
-   **ML Library (`torch`):** PyTorch is the core engine for all machine learning operations. It's used to load model state dictionaries, inspect tensors, and perform manipulations like quantization and conversion.
-   **Serialization (`safetensors`):** The `safetensors` library is used for safe and fast loading/saving of model weights, providing a secure alternative to Python's pickle format used in `.pt` files.

### 3. Communication Protocol

The two components communicate over local HTTP.

1.  The Go frontend packages user requests (e.g., operation type, file path, options) into a JSON payload.
2.  It sends a `POST` (or `GET` for simple queries like `/check`) request to the appropriate endpoint on the Python backend (e.g., `http://127.0.0.1:8000/quantize`).
3.  The Python backend processes the request, performs the model operation, and returns a JSON response containing the result (`success`, `message`, `data`, etc.).
4.  The Go frontend parses this JSON response and displays the information to the user in a formatted way, either in the TUI or on the command line.

