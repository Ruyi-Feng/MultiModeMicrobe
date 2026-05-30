import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch
import yaml

from exp.data_provider import ESM2Representation
from exp.e2eprediction import build_e2e_model, get_property_list, load_aa_encoder
from model import E2EPrediction
from model.property_encoder import PROPERTY_RANGES, denormalize_property


def load_yaml_config(yml_path: str) -> dict:
    with open(yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if cfg is None:
        raise ValueError(f"Empty config file: {yml_path}")
    return cfg


def find_faa_files(temp_compare_dir: str) -> list[Path]:
    base = Path(temp_compare_dir)
    if not base.exists():
        raise FileNotFoundError(f"temp_compare folder not found: {temp_compare_dir}")
    faa_files = sorted(base.glob("*.faa"))
    if len(faa_files) < 2:
        raise ValueError(
            f"Need at least 2 .faa files in {temp_compare_dir}, found {len(faa_files)}"
        )
    return faa_files


def build_esm_args(cfg: dict, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        cut_off=cfg.get("cut_off", 1024),
        model_name=cfg.get("model_name", "esm2_t12_35M_UR50D"),
        batch_size=cfg.get("batch_size", 2),
        device=device,
    )


def extract_h5_for_faa(
    extractor: ESM2Representation,
    faa_path: Path,
    h5_output_dir: str,
    overwrite: bool,
) -> Path:
    sample_id = faa_path.stem
    h5_path = Path(h5_output_dir) / f"{sample_id}.h5"
    if h5_path.exists() and not overwrite:
        return h5_path

    protein_index = {sample_id: []}
    extractor.get_individual_representation(
        protein_path=str(faa_path),
        save_path=h5_output_dir,
        max_shape=(None, extractor.dim),
        protein_index=protein_index,
        bacdive_id=sample_id,
    )
    return h5_path


def build_model_from_cfg(cfg: dict, checkpoint_path: str, device: str) -> E2EPrediction:
    args = SimpleNamespace(**cfg)
    args.device = device
    aa_encoder = load_aa_encoder(args)
    model = build_e2e_model(args, aa_encoder).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict_temperature_for_h5(
    model: E2EPrediction,
    h5_path: Path,
    device: str,
    temp_index: int,
    max_seq_len: int | None = None,
) -> tuple[float, float]:
    with h5py.File(h5_path, "r") as f:
        aa_repr = f["protein_features"][:]
    aa_repr = torch.tensor(aa_repr, dtype=torch.float32)
    if aa_repr.ndim == 1:
        aa_repr = aa_repr.unsqueeze(0)
    if max_seq_len is not None and aa_repr.size(0) > max_seq_len:
        aa_repr = aa_repr[:max_seq_len]

    aa_seq = aa_repr.unsqueeze(0).to(device)
    padding_mask = torch.ones((1, aa_repr.size(0)), dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(aa_seq, padding_mask=padding_mask)
        pred = out["pred"][0]
    temp_norm = float(pred[temp_index].item())
    # 用统一的 PROPERTY_RANGES 反归一化，避免旧的 ×100 假设
    temp_celsius = float(denormalize_property(torch.tensor(temp_norm), "temp").item())
    return temp_norm, temp_celsius


def main():
    parser = argparse.ArgumentParser(
        description="Compare temperature predictions for masked/unmasked faa files."
    )
    parser.add_argument(
        "--temp_compare_dir",
        type=str,
        default="./temp_compare",
        help="Folder containing the faa files to compare.",
    )
    parser.add_argument(
        "--config_yml",
        type=str,
        default="./config/scripts/e2e_prediction_temp.yml",
        help="E2E model config yml path. If None, auto-detect.",
    )
    parser.add_argument(
        "--h5_output_dir",
        type=str,
        default=None,
        help="Output folder for extracted h5 files. Default: <temp_compare_dir>/h5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--overwrite_h5",
        action="store_true",
        help="Overwrite existing h5 files.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config_yml)
    checkpoint_path = cfg.get("resume")
    if checkpoint_path is None:
        raise ValueError("Checkpoint not provided and `resume` is missing in config yml.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    temp_compare_dir = os.path.abspath(args.temp_compare_dir)
    h5_output_dir = (
        os.path.abspath(args.h5_output_dir)
        if args.h5_output_dir
        else os.path.join(temp_compare_dir, "h5")
    )
    os.makedirs(h5_output_dir, exist_ok=True)

    faa_files = find_faa_files(temp_compare_dir)
    esm_args = build_esm_args(cfg, args.device)
    extractor = ESM2Representation(esm_args)

    h5_files = []
    for faa_path in faa_files:
        h5_path = extract_h5_for_faa(
            extractor=extractor,
            faa_path=faa_path,
            h5_output_dir=h5_output_dir,
            overwrite=args.overwrite_h5,
        )
        h5_files.append(h5_path)

    model = build_model_from_cfg(cfg=cfg, checkpoint_path=checkpoint_path, device=args.device)
    max_seq_len = cfg.get("max_seq_len", None)

    # 按训练时的 property_list 解析温度位置，不能硬编码 index
    cfg_ns = SimpleNamespace(**cfg)
    property_list = get_property_list(cfg_ns)
    if "temp" not in property_list:
        raise ValueError(
            f"temp must be enabled in config (use_temp: True). Got property_list={property_list}"
        )
    temp_index = property_list.index("temp")

    print(f"\nLoaded config: {args.config_yml}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Property list: {property_list}, temp_index={temp_index}, temp_range={PROPERTY_RANGES['temp']}")
    print(f"Using faa files in: {temp_compare_dir}\n")
    for faa_path, h5_path in zip(faa_files, h5_files):
        temp_norm, temp_c = predict_temperature_for_h5(
            model=model,
            h5_path=h5_path,
            device=args.device,
            temp_index=temp_index,
            max_seq_len=max_seq_len,
        )
        print(
            f"{faa_path.name}: temp_norm={temp_norm:.6f}, "
            f"temp_pred={temp_c:.2f} C, h5={h5_path}"
        )


if __name__ == "__main__":
    main()