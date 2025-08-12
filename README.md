# Nexa ToolKit

**Nexa ToolKit** is a high-performance, cross-platform command-line tool designed to simplify common machine learning model operations. Featuring a beautiful terminal user interface (TUI) and a powerful backend, it makes complex tasks like quantization, pruning, and format conversion accessible to everyone.

[![Watch the video](https://github.com/DarkStarStrix/NexaToolKit/blob/master/Screenshot%202025-08-12%20164253.png)](https://github.com/DarkStarStrix/NexaToolKit/blob/master/NexaToolKit%20Service%20-%20Made%20with%20Clipchamp.mp4)

## Why Nexa ToolKit?

Are you tired of writing and maintaining one-off Python scripts for basic MLOps tasks?

-   **Searching for old scripts**: "Where is that conversion script I wrote three months ago?"
-   **Dependency nightmares**: "Why did this script break after a library update?"
-   **Inconsistent results**: "Did I use the same quantization settings as last time?"

Nexa ToolKit solves this by providing a **standardized, reliable, and low-friction tool** for all your common model operations. It's built for developers who want to move fast and ensure their workflows are scalable and repeatable.

## Features

-   **Intuitive TUI**: A user-friendly terminal interface guides you through every operation.
-   **Direct CLI Mode**: Power users can script and automate workflows using direct CLI commands.
-   **Model Analysis**: Get detailed statistics about your model, including parameter count, memory size, and a breakdown of layer types.
-   **Format Conversion**: Seamlessly convert between `.pt`, `.pth`, and `.safetensors`, with support for sharded models.
-   **Quantization (Starter & Pro)**: Reduce model size and improve performance with `fp16`, `bf16`, and `int8` quantization.
-   **Advanced Pruning (Pro)**: Implement `unstructured` (weight-based) and `structured` (filter/channel-based) pruning to create sparse models.
-   **Model Merging (Pro)**: Average the weights of multiple models to create powerful ensembles.
-   **Benchmarking (Pro)**: Measure model latency and throughput with configurable inputs.
-   **Evaluation Suite (Pro)**: Run custom evaluation scripts against your models.
-   **Cross-Platform**: The Go-based frontend runs natively on Windows, macOS, and Linux.

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   Go (1.18+) for compiling the frontend.
-   A valid **License Key** purchased from our website.

### Installation

After purchasing, you will receive an `install.sh` and `launch.sh` script.

1.  **Run the installer:**
    This will check your dependencies, clone the repository, and set up your environment.
    ```bash
    ./install.sh
    ```

2.  **Launch the application:**
    Use this script every time you want to run Nexa ToolKit. It starts the required backend services and launches the TUI.
    ```bash
    ./launch.sh
    ```

You can also provide a model file directly to start with it loaded:

```bash
./launch.sh C:\path\to\your\model.safetensors
```

## Evaluation Suite

Nexa ToolKit allows you to run custom Python evaluation scripts against your models.

**Security Warning**: This feature executes arbitrary Python code from a file you provide. It is extremely powerful but carries significant security risks. **Only run evaluation scripts from sources you trust, and ideally, run the entire application in a sandboxed environment like Docker.**

### How it Works

1.  Create a Python script (e.g., `my_eval.py`).
2.  Inside the script, you will have access to a `model` object (the loaded `torch.nn.Module`) and a `results` dictionary.
3.  Perform your evaluation and store your findings in the `results` dictionary.

**Example `my_eval.py`:**
```python
# my_eval.py
# This script checks if the model can perform a forward pass and reports the output shape.

try:
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224) # Adjust shape as needed
    
    # Perform a forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    # Store results
    results['status'] = 'Success'
    results['output_shape'] = list(output.shape)
    results['output_sample'] = str(output.flatten()[:5].tolist())

except Exception as e:
    results['status'] = 'Error'
    results['error_message'] = str(e)

```

You can then call the `/eval` endpoint via the CLI or TUI, providing the path to your model and `my_eval.py`.

## Self-Hosting with Docker Compose (Recommended)

For a stable, isolated, and production-ready environment, run the backend using Docker Compose. This setup includes an Nginx reverse proxy to sandbox the backend server, handle HTTPS, and add security headers.

### For Development

This mode enables hot-reloading for the backend. Any changes you make to the Python code will automatically restart the server.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/nexa-toolkit.git
    cd nexa-toolkit
    ```

2.  **Configure Environment:**
    -   Copy the example environment file: `cp .env.example .env`
    -   Edit the `.env` file. Ensure `DEV_MODE=true` is set. Add your Stripe test keys.

3.  **Build and Run:**
    ```bash
    docker-compose up --build -d
    ```
    The backend is now running and accessible via `https://localhost`. Since it uses a self-signed certificate, you may need to accept a browser security warning.

### For Production

For a production deployment, you should use valid SSL certificates and disable development mode.

1.  **Configure Environment:**
    -   Edit your `.env` file and set `DEV_MODE=false`.
    -   Add your live Stripe keys.
    -   **Developer Key**: You can set a permanent, strong, secret `DEV_API_KEY`. If you leave it unset, a secure, random key will be generated on the first run and printed to the container logs. You must use this key to access the dev sandbox.

2.  **SSL Certificates:**
    -   Replace the self-signed certificates generated in `nginx/Dockerfile` with your own valid certificates. You can do this by mounting a volume with your certs in `docker-compose.yml`.

3.  **Build and Run:**
    ```bash
    docker-compose up --build -d
    ```

### Managing the Services

-   **View logs:** `docker-compose logs -f`
-   **Stop the services:** `docker-compose down`

### Developer Sandbox

The backend includes a sandboxed environment for developers at the `/dev` route. This allows you to access special administrative endpoints that are protected by a separate `DEV_API_KEY`.

To use it, set `DEV_API_KEY` in your `.env` file and send requests to `/dev/...` with an `X-Dev-Key` header containing your key. In a production environment, if you do not set a key, check the server logs on first startup to retrieve the auto-generated key.

**Example: Generating a new license key**
```bash
curl -X POST https://localhost/dev/generate-license \
  -H "X-Dev-Key: your-secret-developer-key"
```

For added security, you can configure the `nginx/nginx.conf` file to only allow requests to the `/dev/` path from your specific IP address.

## ⚖️ License

The core source code of this project is licensed under the **Apache License 2.0**. Please see the [LICENSE](LICENSE) file for details.

Usage of the compiled application requires a commercial license, which can be purchased on our website.
