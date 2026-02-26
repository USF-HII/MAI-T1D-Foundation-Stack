# foundation_adapter.py
import torch
from transformers import AutoModel, AutoConfig
import joblib
import numpy as np
from accelerate import Accelerator

class FoundationAdapter:
    """
    Universal adapter for any foundation model.
    Works on CPU, local GPU (RTX etc.), or H200 with zero code changes.
    """
    def __init__(
        self,
        model_path: str,
        modality: str,
        device: str = "auto",          # "auto", "cpu", "cuda", "cuda:0"
        low_memory: bool = False       # set True on small GPUs
    ):
        self.modality = modality
        self.model_path = model_path

        # ---------- Device & dtype logic ----------
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.low_memory = low_memory

        # Smart dtype selection
        if self.device == "cpu":
            self.torch_dtype = torch.float32
        else:
            # H200 / Ampere+ → bf16 (fastest & most stable)
            if torch.cuda.get_device_capability(0)[0] >= 8:
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

        # ---------- Load model ----------
        if any(x in model_path.lower() for x in ["geneformer", "scgpt", "dnabert", "enformer", "hyena"]):
            # Hugging Face foundation models
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if self.device != "cpu" and not low_memory:
                load_kwargs["device_map"] = "auto"   # H200 / large GPU magic
            else:
                load_kwargs["device_map"] = None

            self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
            if self.device != "cpu" and "device_map" not in load_kwargs:
                self.model = self.model.to(self.device)

        elif model_path.endswith(('.pt', '.pth', '.bin')):
            # Raw PyTorch checkpoint
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            self.model.eval()

        elif model_path.endswith(('.pkl', '.joblib')):
            # Tabular / sklearn (TabPFN, XGBoost, etc.)
            self.model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        self.model.eval()

    def predict_and_extract(self, batch):
        """Returns: probs, logits (optional), embeddings"""
        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module):  # Foundation models
                # Move batch to correct device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device) if torch.is_tensor(batch) else batch

                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                embeddings = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs
                logits = getattr(outputs, 'logits', None)
                probs = torch.softmax(logits, dim=-1) if logits is not None else None
            else:
                # sklearn-style models
                probs = self.model.predict_proba(batch)
                logits = None
                embeddings = np.array(probs)

            return (
                probs.cpu().numpy() if torch.is_tensor(probs) else probs,
                logits.cpu().numpy() if torch.is_tensor(logits) else logits,
                embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
            )
