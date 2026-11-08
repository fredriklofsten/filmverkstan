"""Känner av grafikkortet och föreslår modell."""


def info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "name": "PyTorch saknas", "vram_gb": 0.0}
    if not torch.cuda.is_available():
        return {"cuda": False, "name": "Inget CUDA-grafikkort hittades", "vram_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": props.name, "vram_gb": round(props.total_memory / 1024**3, 1)}


def recommended_model(vram_gb: float) -> str:
    # Wan 2.2 5B ger bäst kvalitet men vill ha mycket minne.
    # LTX-Video är snabbare och klarar sig på mindre kort.
    return "wan" if vram_gb >= 20 else "ltx"


def summary() -> str:
    i = info()
    if not i["cuda"]:
        return f"⚠️ {i['name']}. Videogenerering kräver ett Nvidia-kort med CUDA."
    rec = "Wan 2.2 (bäst kvalitet)" if recommended_model(i["vram_gb"]) == "wan" else "LTX-Video (snabb, sparsam)"
    return f"✅ {i['name']} – {i['vram_gb']} GB videominne. Rekommenderad modell: {rec}."
