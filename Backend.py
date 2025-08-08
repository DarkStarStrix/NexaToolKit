#!/usr/bin/env python3
"""
Nexa Toolkit Backend - Model Processing Engine
High-performance ML model operations exposed via a FastAPI server.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import stripe
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from safetensors.torch import save_file, load_file
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from passkeys import server
from passkeys.storage import CredentialStorage, MemoryStorage

# --- Configuration & Initialization ---
load_dotenv()

is_dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

# Load secrets from environment variables
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
# In a real app, you would have a database of customers and their keys
VALID_API_KEYS = {"your-generated-api-key-for-testing"}
DEV_API_KEY = os.getenv("DEV_API_KEY")

# When in production, if no dev key is set, generate a secure one and log it.
# This ensures the dev endpoint is always protected, and the key is only visible to the server admin.
if not is_dev_mode and not DEV_API_KEY:
    DEV_API_KEY = secrets.token_urlsafe(32)
    logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logging.warning("!!! NO DEV_API_KEY SET IN PRODUCTION MODE                   !!!")
    logging.warning(f"!!! Generated a temporary developer key: {DEV_API_KEY} !!!")
    logging.warning("!!! SET THIS IN YOUR .env FILE TO PERSIST ACROSS RESTARTS   !!!")
    logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app, rate limiter, and Stripe
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Nexa ToolKit Backend")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

stripe.api_key = STRIPE_API_KEY

# --- Passkey & User Auth Setup ---
# In a real app, you would use a persistent database like SQLite or PostgreSQL.
# For this example, we use in-memory storage.
user_storage: Dict[str, Dict[str, Any]] = {}
credential_storage: CredentialStorage = MemoryStorage()

class User(BaseModel):
    username: str
    id: str
    credentials: list = []

# --- Licensing & Feature Gating ---
PUBLIC_KEY = "NEXA_PUBLIC_KEY_FOR_DEMO"

def get_license_tier(x_license_key: str = Header(...)) -> str:
    """Dependency that verifies a license and returns its tier."""
    # In dev mode, allow the DEV_API_KEY to act as a pro-level key for easy testing.
    if is_dev_mode and DEV_API_KEY and hmac.compare_digest(x_license_key, DEV_API_KEY):
        return "pro"

    try:
        message, signature = x_license_key.rsplit('.', 1)
        expected_signature = hmac.new(
            PUBLIC_KEY.encode('utf-8'), msg=message.encode('utf-8'), digestmod=hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_signature, signature):
            raise ValueError("Invalid signature")

        license_data_str = base64.urlsafe_b64decode(message).decode('utf-8')
        license_data = dict(item.split(":") for item in license_data_str.split(","))

        # Example: expiry check
        # if int(license_data.get("expiry", 0)) < int(time.time()):
        #     raise ValueError("License expired")

        return license_data.get("tier", "none")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid or missing license key.")

def require_pro_tier(tier: str = Depends(get_license_tier)):
    """Dependency that requires a 'pro' or 'lifetime' license tier."""
    if tier not in ["pro", "lifetime"]:
        raise HTTPException(status_code=403, detail="This feature requires a Pro or Lifetime license.")

def require_starter_tier(tier: str = Depends(get_license_tier)):
    """Dependency that requires at least a 'starter' license."""
    if tier not in ["starter", "pro", "lifetime"]:
        raise HTTPException(status_code=403, detail="This feature requires a Starter, Pro, or Lifetime license.")

def verify_dev_key(x_dev_key: str = Header(...)):
    """Dependency to verify the developer API key."""
    if not DEV_API_KEY:
        raise HTTPException(status_code=500, detail="Developer key not configured on server.")
    if not hmac.compare_digest(x_dev_key, DEV_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid developer key.")


# --- Data Models ---

class ModelInfo(BaseModel):
    """Data class for model metadata and statistics"""
    format_type: str = ""
    file_size: int = 0
    memory_size: int = 0
    num_parameters: int = 0
    num_tensors: int = 0
    tensor_info: list = []
    architecture: str = "Unknown"
    layer_type_counts: Dict[str, int] = {}
    dtype: str = ""

class OperationResult(BaseModel):
    """Standard response format for all operations"""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    execution_time: float = 0.0
    error: Optional[str] = None

# --- API Request Models ---

class FilePathRequest(BaseModel):
    """Request model for file path based operations"""
    file_path: str

class ConvertRequest(BaseModel):
    """Request model for converting model formats"""
    file_path: str
    target_format: str = "safetensors"
    output_path: Optional[str] = None

class QuantizeRequest(BaseModel):
    """Request model for quantizing models"""
    file_path: str
    method: str = "fp16"
    output_path: Optional[str] = None

class PruneRequest(BaseModel):
    """Request model for pruning models"""
    file_path: str
    amount: float = Field(0.2, gt=0, lt=1)
    pruning_type: str = "unstructured" # 'unstructured' or 'structured'
    output_path: Optional[str] = None

class MergeRequest(BaseModel):
    """Request model for merging models"""
    file_paths: List[str]
    output_path: Optional[str] = None

class BenchmarkRequest(BaseModel):
    """Request model for benchmarking"""
    file_path: str
    input_shape: List[int] = [1, 3, 224, 224] # Default to a common image size
    num_runs: int = 50

class EvalRequest(BaseModel):
    """Request model for running an evaluation script"""
    model_path: str
    eval_script_path: str

# --- Core Logic ---

class ModelProcessor:
    """Core model processing engine, refactored for clarity and efficiency."""

    def __init__(self):
        """Initializes the ModelProcessor."""
        pass

    @staticmethod
    def _generate_license_key(tier: str = "starter") -> str:
        """Generates a new, valid license key for a specific tier."""
        # This would contain expiry, features, etc.
        expiry = int(time.time() + 3600 * 24 * 365 * 5) # 5-year expiry for demo
        license_data = f"tier:{tier},expiry:{expiry}"
        message = base64.urlsafe_b64encode(license_data.encode('utf-8')).decode('utf-8')
        signature = hmac.new(
            PUBLIC_KEY.encode('utf-8'),
            msg=message.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        return f"{message}.{signature}"

    @staticmethod
    def _load_state_dict_or_model(file_path: Path, load_full_model: bool = False) -> Union[Dict[str, torch.Tensor], torch.nn.Module]:
        """
        Loads a model's state dictionary or the full model from a file.
        Now supports loading sharded .safetensors models if a directory or index file is provided.
        """
        if not file_path or not file_path.name:
            raise ValueError("A valid file path must be provided.")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Handle sharded safetensors
        index_path = None
        if file_path.is_dir():
            potential_index = file_path / "model.safetensors.index.json"
            if potential_index.exists():
                index_path = potential_index
        elif file_path.name.endswith(".index.json"):
            index_path = file_path

        if index_path:
            if load_full_model:
                raise ValueError("Cannot load a full torch.nn.Module from sharded .safetensors, only a state_dict.")

            state_dict = {}
            with open(index_path, 'r') as f:
                index = json.load(f)

            shard_files = {
                shard_path: load_file(index_path.parent / shard_path)
                for shard_path in set(index["weight_map"].values())
            }

            for tensor_name, shard_path in index["weight_map"].items():
                state_dict[tensor_name] = shard_files[shard_path][tensor_name]
            return state_dict

        source_format = file_path.suffix.lower()
        if source_format not in ['.pt', '.pth', '.safetensors']:
            raise ValueError(f"Unsupported format: {source_format}")

        if source_format == '.safetensors':
            if load_full_model:
                raise ValueError("Cannot load a full torch.nn.Module from .safetensors, only a state_dict.")
            return load_file(file_path)

        loaded_object = torch.load(file_path, map_location='cpu', weights_only=False)

        if isinstance(loaded_object, torch.nn.Module):
            return loaded_object if load_full_model else loaded_object.state_dict()

        if isinstance(loaded_object, dict):
            if load_full_model:
                raise ValueError("Loaded a state_dict but a full model object was required.")
            for key in ['model_state_dict', 'state_dict']:
                if key in loaded_object and isinstance(loaded_object[key], dict):
                    return loaded_object[key]
            return loaded_object

        raise TypeError(f"Unsupported object type loaded from {file_path}: {type(loaded_object)}")

    def analyze_model(self, file_path_str: str) -> ModelInfo:
        """Extracts comprehensive model statistics and metadata."""
        file_path = Path(file_path_str)
        state_dict = self._load_state_dict_or_model(file_path)

        model_info = ModelInfo()
        model_info.file_size = file_path.stat().st_size
        model_info.format_type = file_path.suffix.lower()

        total_params = 0
        memory_size = 0
        tensor_infos = []
        layer_type_counts = {}

        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                numel = tensor.numel()
                tensor_infos.append({
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size": numel
                })
                total_params += numel
                memory_size += tensor.element_size() * numel

                # Enhanced architecture analysis
                if 'conv' in name: layer_type_counts['conv'] = layer_type_counts.get('conv', 0) + 1
                elif 'attention' in name: layer_type_counts['attention'] = layer_type_counts.get('attention', 0) + 1
                elif 'ln' in name or 'layernorm' in name: layer_type_counts['norm'] = layer_type_counts.get('norm', 0) + 1
                elif 'mlp' in name: layer_type_counts['mlp'] = layer_type_counts.get('mlp', 0) + 1
                elif 'embedding' in name: layer_type_counts['embedding'] = layer_type_counts.get('embedding', 0) + 1


        model_info.num_parameters = total_params
        model_info.num_tensors = len(tensor_infos)
        model_info.memory_size = memory_size
        model_info.tensor_info = tensor_infos
        model_info.layer_type_counts = layer_type_counts

        layer_names = list(state_dict.keys())
        if layer_type_counts.get('attention', 0) > 0:
            model_info.architecture = "Transformer-based"
        elif layer_type_counts.get('conv', 0) > 0:
            model_info.architecture = "CNN-based"
        elif any('resnet' in name.lower() for name in layer_names):
            model_info.architecture = "ResNet"
        elif any('bert' in name.lower() for name in layer_names):
            model_info.architecture = "BERT"

        if tensor_infos:
            model_info.dtype = tensor_infos[0]['dtype']

        return model_info

    def convert_model(self, file_path_str: str, target_format: str, output_path_str: Optional[str]) -> dict:
        """Converts a model between supported formats."""
        file_path = Path(file_path_str)
        state_dict = self._load_state_dict_or_model(file_path)

        if output_path_str is None:
            suffix = '.safetensors' if target_format == 'safetensors' else '.pt'
            output_path = file_path.with_suffix(suffix)
        else:
            output_path = Path(output_path_str)

        if target_format == "safetensors":
            save_file(state_dict, output_path)
        elif target_format == "pytorch":
            torch.save(state_dict, output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        return {
            "input_file": file_path_str,
            "output_file": str(output_path),
            "source_format": file_path.suffix.lower(),
            "target_format": target_format,
            "output_size": output_path.stat().st_size
        }

    def quantize_model(self, file_path_str: str, method: str, output_path_str: Optional[str]) -> dict:
        """Quantizes model weights to reduce memory usage."""
        file_path = Path(file_path_str)
        state_dict = self._load_state_dict_or_model(file_path)

        if output_path_str is None:
            output_path = file_path.with_suffix(f'.{method}{file_path.suffix}')
        else:
            output_path = Path(output_path_str)

        quantized_state_dict = {}
        original_size, quantized_size = 0, 0
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        target_dtype = dtype_map.get(method)

        if not target_dtype and method != "int8":
            raise ValueError(f"Unsupported quantization method: {method}")

        for name, tensor in state_dict.items():
            original_size += tensor.element_size() * tensor.numel()
            if tensor.is_floating_point():
                if method == "int8":
                    quantized_tensor = tensor.mul(127).round().clamp(-128, 127).to(torch.int8)
                else:
                    quantized_tensor = tensor.to(target_dtype)
            else:
                quantized_tensor = tensor
            quantized_state_dict[name] = quantized_tensor
            quantized_size += quantized_tensor.element_size() * quantized_tensor.numel()

        save_file(quantized_state_dict, output_path)

        return {
            "input_file": file_path_str,
            "output_file": str(output_path),
            "method": method,
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 1.0,
            "space_saved": original_size - quantized_size
        }

    def prune_model(self, file_path_str: str, amount: float, pruning_type: str, output_path_str: Optional[str]) -> dict:
        """Prunes model weights using specified method."""
        if pruning_type == "unstructured":
            return self._unstructured_prune(file_path_str, amount, output_path_str)
        elif pruning_type == "structured":
            return self._structured_prune(file_path_str, amount, output_path_str)
        else:
            raise ValueError(f"Unsupported pruning type: {pruning_type}")

    def _unstructured_prune(self, file_path_str: str, amount: float, output_path_str: Optional[str]) -> dict:
        """Prunes model weights by zeroing out the lowest‐magnitude entries."""
        file_path = Path(file_path_str)
        state_dict = self._load_state_dict_or_model(file_path, load_full_model=False)

        pruned_state = {}
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 1 and tensor.is_floating_point():
                flat = tensor.abs().view(-1)
                threshold = float(torch.quantile(flat, amount))
                mask = torch.abs(tensor) >= threshold
                pruned_tensor = tensor * mask.to(tensor.dtype)
                pruned_state[name] = pruned_tensor
            else:
                pruned_state[name] = tensor

        out_path = self._get_output_path(file_path, output_path_str, f"_pruned{int(amount*100)}")
        self._save_model(pruned_state, out_path)

        return {
            "input_file": file_path_str,
            "output_file": str(out_path),
            "pruning_amount": amount,
            "pruning_type": "unstructured"
        }

    def _structured_prune(self, file_path_str: str, amount: float, output_path_str: Optional[str]) -> dict:
        """Prunes entire channels/filters based on L1 norm."""
        file_path = Path(file_path_str)
        state_dict = self._load_state_dict_or_model(file_path, load_full_model=False)

        pruned_state = state_dict.copy()
        for name, tensor in state_dict.items():
            # Target convolutional or linear layers for pruning
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and ("weight" in name):
                # Sum of absolute values across channels (dim 1) for each filter (dim 0)
                l1_norm = torch.sum(torch.abs(tensor), dim=tuple(range(1, tensor.ndim)))

                num_to_prune = int(len(l1_norm) * amount)
                if num_to_prune > 0:
                    # Find the indices of the filters with the smallest L1 norms
                    pruning_indices = torch.topk(l1_norm, num_to_prune, largest=False).indices

                    # Create a mask to zero out the pruned filters
                    mask = torch.ones_like(l1_norm, dtype=torch.bool)
                    mask[pruning_indices] = False

                    # Apply mask. Unsqueeze to match tensor dimensions for broadcasting.
                    mask_shape = [len(l1_norm)] + [1] * (tensor.ndim - 1)
                    pruned_state[name] = tensor * mask.view(mask_shape)

        out_path = self._get_output_path(file_path, output_path_str, f"_pruned_struct_{int(amount*100)}")
        self._save_model(pruned_state, out_path)

        return {
            "input_file": file_path_str,
            "output_file": str(out_path),
            "pruning_amount": amount,
            "pruning_type": "structured"
        }

    def merge_models(self, file_paths: List[str], output_path_str: Optional[str]) -> dict:
        """Merges the state dicts of multiple models by averaging their weights."""
        if len(file_paths) < 2:
            raise ValueError("At least two models are required for merging.")

        # Load all state dicts
        state_dicts = [self._load_state_dict_or_model(Path(fp)) for fp in file_paths]

        # Use the first model's keys as the reference
        merged_state_dict = {}
        ref_keys = state_dicts[0].keys()

        for key in ref_keys:
            tensors_to_merge = [sd[key] for sd in state_dicts if key in sd and isinstance(sd[key], torch.Tensor)]
            if tensors_to_merge and all(t.shape == tensors_to_merge[0].shape for t in tensors_to_merge):
                # Stack and average the tensors
                merged_tensor = torch.mean(torch.stack(tensors_to_merge), dim=0)
                merged_state_dict[key] = merged_tensor
            else:
                # If not present in all models or shapes mismatch, take from the first model
                merged_state_dict[key] = state_dicts[0][key]

        # Determine output path
        first_path = Path(file_paths[0])
        out_path = self._get_output_path(first_path, output_path_str, "_merged")
        self._save_model(merged_state_dict, out_path)

        return {
            "output_file": str(out_path),
            "models_merged": len(file_paths),
            "output_size": out_path.stat().st_size
        }

    @staticmethod
    def benchmark_model(file_path_str: str, input_shape: List[int], num_runs: int) -> dict:
        """Performs inference benchmark on a model."""
        file_path = Path(file_path_str)

        try:
            model = ModelProcessor._load_state_dict_or_model(file_path, load_full_model=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Benchmarking requires a full torch.nn.Module object. Failed to load from {file_path_str}. Error: {e}")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Benchmarking requires a full torch.nn.Module object, but loaded a state_dict.")

        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        try:
            dummy_input = torch.randn(*input_shape, device=device)
        except Exception as e:
            raise ValueError(f"Failed to create dummy input with shape {input_shape}. Error: {e}")

        # Warm-up runs
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)

        # Timed runs
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        end_time = time.time()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / num_runs) * 1000
        throughput_fps = num_runs / total_time

        return {
            "device": device,
            "num_runs": num_runs,
            "input_shape": input_shape,
            "avg_latency_ms": round(avg_latency_ms, 4),
            "throughput_fps": round(throughput_fps, 2)
        }

    def run_evaluation(self, model_path: str, eval_script_path: str) -> dict:
        """
        Runs a user-provided evaluation script against a model.
        WARNING: This method executes arbitrary code and poses a significant security risk.
        It should only be used in a sandboxed environment with trusted scripts.
        """
        model_file = Path(model_path)
        eval_script = Path(eval_script_path)

        if not eval_script.exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script_path}")

        try:
            model = self._load_state_dict_or_model(model_file, load_full_model=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Evaluation requires a full torch.nn.Module object. Failed to load from {model_path}. Error: {e}")

        with open(eval_script, 'r') as f:
            script_code = f.read()

        # Prepare a scope for the script to run in
        eval_scope = {
            'model': model,
            'torch': torch,
            'results': {} # A dict to store results
        }

        # Execute the script
        exec(script_code, eval_scope)

        # Return whatever the script put into the 'results' dictionary
        return eval_scope.get('results', {"message": "Evaluation script ran, but no 'results' dictionary was found."})


    @staticmethod
    def check_environment() -> dict:
        """Validates environment and dependencies."""
        dependencies = {'torch': {'available': True, 'version': torch.__version__}}
        try:
            import safetensors
            dependencies['safetensors'] = {'available': True, 'version': safetensors.__version__}
        except ImportError:
            dependencies['safetensors'] = {'available': False}
        return {'dependencies': dependencies, 'python_version': sys.version}

    @staticmethod
    def _get_output_path(input_path: Path, output_path_str: Optional[str], suffix_mod: str) -> Path:
        if output_path_str:
            return Path(output_path_str)
        return input_path.with_name(f"{input_path.stem}{suffix_mod}{input_path.suffix}")

    @staticmethod
    def _save_model(state_dict: Dict[str, torch.Tensor], out_path: Path):
        if out_path.suffix.lower() == ".safetensors":
            save_file(state_dict, out_path)
        else:
            torch.save(state_dict, out_path)

# --- Security Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Custom Exception Handling ---
@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    logging.error(f"An unhandled error occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )

# --- Helper Functions ---
async def fulfill_order(session):
    """
    Handles post-payment logic after a successful Stripe checkout session.
    This function now correctly generates a signature-based license key.
    """
    customer_email = session.get("customer_details", {}).get("email")
    subscription_id = session.get("subscription")
    logging.info(f"Fulfilling order for customer: {customer_email} with subscription: {subscription_id}")

    # This is a placeholder. In a real app, you would look up the price ID
    # from the session to determine which product was purchased.
    # For now, we assume a "pro" tier purchase.
    # TODO: Map session.line_items to your product tiers.
    tier = "pro" # Default to pro for this example

    # Use the same method as the dev endpoint to generate a valid, signed key.
    new_key = ModelProcessor._generate_license_key(tier=tier)
    # In a real app, you would save this key to your database, associated with the customer,
    # and then email it to them.
    logging.info(f"Generated '{tier}' license key for {customer_email}: {new_key}")

# --- FastAPI Application ---

app = FastAPI(title="Nexa Toolkit Backend")
processor = ModelProcessor()

# Routers for different levels of access
public_router = APIRouter()
# Starter features are available to all valid license holders
starter_router = APIRouter(dependencies=[Depends(require_starter_tier)])
# Pro features require a pro or lifetime license
pro_router = APIRouter(dependencies=[Depends(require_pro_tier)])
dev_router = APIRouter(prefix="/dev", dependencies=[Depends(verify_dev_key)])


def run_operation(operation_func, **kwargs) -> OperationResult:
    """Wraps processor methods to handle exceptions and format responses."""
    start_time = time.time()
    try:
        data = operation_func(**kwargs)
        execution_time = time.time() - start_time
        return OperationResult(
            message=f"{operation_func.__name__} completed successfully.",
            data=data,
            execution_time=execution_time
        )
    except (FileNotFoundError, ValueError, TypeError, Exception) as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=400,
            detail=OperationResult(
                success=False,
                message=f"Operation failed: {str(e)}",
                error=type(e).__name__,
                execution_time=execution_time
            ).model_dump()
        )

# --- API Endpoints ---

@public_router.get("/", response_model=OperationResult)
def health_check():
    """Health check endpoint."""
    return OperationResult(message="Nexa Toolkit Backend is running.")

# --- Starter Tier Endpoints ---
@starter_router.post("/analyze", response_model=OperationResult)
def analyze(request: FilePathRequest):
    """Analyzes a model file."""
    return run_operation(processor.analyze_model, file_path_str=request.file_path)

@starter_router.post("/convert", response_model=OperationResult)
def convert(request: ConvertRequest):
    """Converts a model file to a different format."""
    return run_operation(processor.convert_model, file_path_str=request.file_path,
                         target_format=request.target_format, output_path_str=request.output_path)

@starter_router.post("/quantize", response_model=OperationResult)
def quantize(request: QuantizeRequest):
    """Quantizes a model file."""
    return run_operation(processor.quantize_model, file_path_str=request.file_path,
                         method=request.method, output_path_str=request.output_path)

# --- Pro Tier Endpoints ---
@pro_router.post("/prune", response_model=OperationResult)
def prune(request: PruneRequest):
    """Prunes a model file."""
    return run_operation(
        processor.prune_model,
        file_path_str=request.file_path,
        amount=request.amount,
        pruning_type=request.pruning_type,
        output_path_str=request.output_path,
    )

@pro_router.post("/merge", response_model=OperationResult)
def merge(request: MergeRequest):
    """Merges multiple model files."""
    return run_operation(
        processor.merge_models,
        file_paths=request.file_paths,
        output_path_str=request.output_path
    )

@pro_router.post("/benchmark", response_model=OperationResult)
def benchmark(request: BenchmarkRequest):
    """Benchmarks a model file."""
    return run_operation(
        processor.benchmark_model,
        file_path_str=request.file_path,
        input_shape=request.input_shape,
        num_runs=request.num_runs
    )

@pro_router.post("/eval", response_model=OperationResult)
def run_eval(request: EvalRequest):
    """
    Runs a custom evaluation script against a model.
    WARNING: This endpoint executes arbitrary Python code provided by the user
    and should only be exposed in a secure, controlled environment.
    """
    return run_operation(
        processor.run_evaluation,
        model_path=request.model_path,
        eval_script_path=request.eval_script_path
    )

@starter_router.get("/check", response_model=OperationResult)
def check_env():
    """Checks the environment for dependencies."""
    return run_operation(processor.check_environment)

# --- Developer Sandbox Endpoint ---
@dev_router.post("/generate-license", response_model=OperationResult)
def generate_license(tier: str = "starter"):
    """Generates a new, valid license key for testing or manual issuance."""
    return run_operation(processor._generate_license_key, tier=tier)


# --- Passkey Authentication Endpoints (Structure Only) ---
@public_router.post("/register-begin")
async def register_begin(username: str):
    """Begin passkey registration"""
    if username in user_storage:
        raise HTTPException(status_code=400, detail="User already exists")

    user_id = secrets.token_hex(16)
    user_storage[username] = {"id": user_id, "credentials": []}

    options, state = server.begin_register(
        user_id=user_id,
        user_name=username,
        user_display_name=username,
        credential_storage=credential_storage,
    )
    # Store state in session or another temporary storage
    # request.session['passkey_state'] = state
    return options

@public_outer.post("/register-complete")
async def register_complete(username: str, request: Request):
    """Complete passkey registration"""
    # state = request.session.get('passkey_state')
    # body = await request.json()
    # user = user_storage.get(username)
    # if not user or not state:
    #     raise HTTPException(status_code=400, detail="Invalid state")
    #
    # credential = server.complete_register(
    #     state=state,
    #     body=body,
    #     credential_storage=credential_storage
    # )
    # user['credentials'].append(credential)
    return {"status": "ok"} # Placeholder

# ... Add /login-begin and /login-complete endpoints similarly ...


@public_router.post("/create-checkout-session")
@limiter.limit("5/minute")
async def create_checkout_session(request: Request, price_id: str):
    """Creates a Stripe Checkout session for a given price ID."""
    if not STRIPE_API_KEY:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=os.getenv("STRIPE_SUCCESS_URL", "http://localhost:8000/success") + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "http://localhost:8000/cancel"),
        )
        return {"checkout_url": checkout_session.url}
    except Exception as e:
        logging.error(f"Stripe session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Could not create checkout session.")

@public_router.post("/stripe-webhook")
@limiter.limit("20/minute")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Handles incoming webhooks from Stripe."""
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook is not configured.")

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=stripe_signature, secret=STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        await fulfill_order(event['data']['object'])
    elif event['type'] in ['invoice.payment_succeeded', 'customer.subscription.updated', 'customer.subscription.deleted']:
        # Handle subscription changes (e.g., renewals, cancellations)
        logging.info(f"Received subscription event: {event['type']}")
        # TODO: Update user's subscription status in your database.
        pass
    else:
        logging.warning(f"Unhandled Stripe event type: {event['type']}")

    return {"status": "success"}

# Add other application endpoints (prune, quantize, etc.) here
# Remember to protect them with an API key check.
@public_router.get("/status")
async def status(request: Request):
    return {"status": "ok"}

# Include all routers in the main application
app.include_router(public_router)
app.include_router(starter_router)
app.include_router(pro_router)
app.include_router(dev_router)

if __name__ == "__main__":
    import uvicorn
    # is_dev_mode is now defined at the top of the file
    # Ensure debug is False for production
    uvicorn.run(
        "Backend:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev_mode
    )
