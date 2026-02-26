# foundation_adapter.py
import torch
from transformers import AutoModel, AutoConfig
import joblib
import numpy as np
from accelerate import Accelerator

class FoundationAdapter:
    """
    Universal adapter for any foundation model (HF, PyTorch, scikit-learn).
    Works seamlessly on H200 (bf16, device_map, accelerate).
    """
    def __init__(self, model_path: str, modality: str, device_map: str = "auto"):
        self.modality = modality
        self.model_path = model_path
        self.accelerator = Accelerator()
        self.device = self.accelerator.device  # H200 picked automatically

        if any(x in model_path.lower() for x in ["geneformer", "scgpt", "dnabert", "enformer", "hyena"]):
            # Hugging Face foundation models
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map
            )
            self.model.eval()
        elif model_path.endswith(('.pt', '.pth', '.bin')):
            # Raw PyTorch checkpoint
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            self.model.eval()
        elif model_path.endswith(('.pkl', '.joblib')):
            # Tabular / sklearn-style (TabPFN, XGBoost, etc.)
            self.model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def predict_and_extract(self, batch):
        """
        Returns: probs, logits (optional), embeddings
        Works for both HF transformers and tabular models.
        """
        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module):  # Foundation models
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                embeddings = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs
                logits = outputs.logits if hasattr(outputs, 'logits') else None
                probs = torch.softmax(logits, dim=-1) if logits is not None else None
            else:  # sklearn-style
                probs = self.model.predict_proba(batch)
                logits = None
                embeddings = np.array(probs)  # fallback

            return (
                probs.cpu().numpy() if torch.is_tensor(probs) else probs,
                logits.cpu().numpy() if torch.is_tensor(logits) else logits,
                embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
            )
