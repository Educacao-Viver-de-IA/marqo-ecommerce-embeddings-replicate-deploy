"""
Marqo E-commerce Embeddings L — modelo SigLIP fine-tunado pra retrieval de produtos.
Retorna embeddings unitários (L2-normalized) que podem ser usados pra busca por similaridade
(cosine = dot product após normalização).
"""
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()
import torch
print(f"[module] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
print(f"[module] imports OK", flush=True)
sys.stdout.flush()

WEIGHTS_DIR = "/src/weights/marqo"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.setup_error = None
        try:
            print(f"[setup] dir: {sorted(os.listdir(WEIGHTS_DIR))[:15]}", flush=True)
        except Exception as e:
            print(f"[setup] err: {e}", flush=True)

        try:
            # Tenta carregar via transformers (mais simples). Marqo é compatível com SigLIP
            from transformers import AutoModel, AutoProcessor
            print(f"[setup] loading via transformers AutoProcessor + AutoModel...", flush=True)
            sys.stdout.flush()
            self.processor = AutoProcessor.from_pretrained(WEIGHTS_DIR, local_files_only=True, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                WEIGHTS_DIR, local_files_only=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            print(f"[setup] DONE via transformers (t={time.time()-t0:.1f}s)", flush=True)
            self.backend = "transformers"
            sys.stdout.flush()
        except Exception as e:
            print(f"[setup] transformers failed ({type(e).__name__}: {e}), tentando open_clip...", flush=True)
            sys.stdout.flush()
            try:
                import open_clip
                # Marqo SigLIP via open_clip
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "hf-hub:Marqo/marqo-ecommerce-embeddings-L",
                    cache_dir=WEIGHTS_DIR,
                )
                self.tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-ecommerce-embeddings-L")
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.backend = "open_clip"
                print(f"[setup] DONE via open_clip (t={time.time()-t0:.1f}s)", flush=True)
            except Exception as e2:
                import traceback
                print(f"[setup] FATAL: {type(e2).__name__}: {e2}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                self.setup_error = f"setup failed: {e2}"

    def predict(
        self,
        image: Path = Input(description="(Opcional) Imagem do produto.", default=None),
        text: str = Input(default="", description="(Opcional) Texto do produto / query."),
        normalize: bool = Input(default=True, description="L2-normaliza embedding."),
    ) -> dict:
        if self.model is None:
            return {"error": f"Modelo não carregou: {getattr(self, 'setup_error', '?')}"}
        if image is None and not text.strip():
            return {"error": "Forneça image ou text."}

        t0 = time.time()
        result = {"backend": self.backend, "image_embedding": None, "text_embedding": None}
        device = next(self.model.parameters()).device

        with torch.inference_mode():
            if image is not None:
                pil = Image.open(image).convert("RGB")
                if self.backend == "transformers":
                    proc_in = self.processor(images=pil, return_tensors="pt")
                    proc_in = {k: v.to(device) for k, v in proc_in.items()}
                    feats = self.model.get_image_features(**proc_in) if hasattr(self.model, 'get_image_features') else self.model(**proc_in).image_embeds
                else:
                    img_t = self.preprocess(pil).unsqueeze(0).to(device)
                    feats = self.model.encode_image(img_t)
                if normalize:
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                result["image_embedding"] = feats[0].float().cpu().tolist()

            if text.strip():
                if self.backend == "transformers":
                    txt_in = self.processor(text=[text], return_tensors="pt", padding=True)
                    txt_in = {k: v.to(device) for k, v in txt_in.items()}
                    tfeats = self.model.get_text_features(**txt_in) if hasattr(self.model, 'get_text_features') else self.model(**txt_in).text_embeds
                else:
                    tokens = self.tokenizer([text]).to(device)
                    tfeats = self.model.encode_text(tokens)
                if normalize:
                    tfeats = tfeats / tfeats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                result["text_embedding"] = tfeats[0].float().cpu().tolist()

        result["embedding_dim"] = len(result["image_embedding"] or result["text_embedding"])
        result["predict_time_s"] = round(time.time() - t0, 3)
        return result
