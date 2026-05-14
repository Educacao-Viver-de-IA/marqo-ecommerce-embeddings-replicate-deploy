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

        # Strategy: usar open_clip com path explícito ao open_clip_pytorch_model.bin
        # e carregar arch do open_clip_config.json. Evita "hf-hub:" prefix que tenta rede.
        try:
            import open_clip
            import json as _json
            cfg_path = os.path.join(WEIGHTS_DIR, "open_clip_config.json")
            weights_path = os.path.join(WEIGHTS_DIR, "open_clip_pytorch_model.bin")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"open_clip_config.json not in {WEIGHTS_DIR}")
            with open(cfg_path) as f:
                oc_cfg = _json.load(f)
            print(f"[setup] open_clip_config keys: {list(oc_cfg.keys())}", flush=True)
            sys.stdout.flush()

            # marqo-fashionSigLIP é variante específica do open_clip
            # Tenta construir do config
            model_cfg = oc_cfg.get("model_cfg", oc_cfg)
            preprocess_cfg = oc_cfg.get("preprocess_cfg", {})

            # Try via factory direct path
            self.model = open_clip.create_model("ViT-L-14", pretrained=None)
            # Load custom weights
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[setup] open_clip state_dict load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
            # Build preprocess transform from config
            self.preprocess = open_clip.image_transform(
                image_size=preprocess_cfg.get("size", 224),
                is_train=False,
                mean=preprocess_cfg.get("mean"),
                std=preprocess_cfg.get("std"),
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.backend = "open_clip"
            print(f"[setup] DONE via open_clip (t={time.time()-t0:.1f}s)", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL open_clip: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

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
