from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from celebv_hq_data import (
    DatasetClipError,
    FaceCentricVideoConfig,
    read_celebv_hq_conditioning_frame,
    read_celebv_hq_video_clip,
)
from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader, Dataset, Sampler


DEFAULT_ROLLOUT_PROMPT_BANK = (
    "A close-up portrait video, the person blinks and slightly turns their head.",
    "A close-up portrait video, the person smiles naturally.",
    "A close-up portrait video, the person talks with subtle facial motion.",
    "A close-up portrait video, the person looks to the side and back.",
    "A close-up portrait video, the person nods gently.",
    "A close-up portrait video, the person laughs softly.",
    "A close-up portrait video, strong wind continuously moves the person's hair.",
    "A close-up portrait video, the camera slowly moves closer.",
    "A close-up portrait video, the background lights flicker softly.",
    "A close-up portrait video, the person changes from a neutral expression to a smile.",
    "A close-up portrait video, strong wind moves the person's hair while the camera slowly moves closer.",
    "A close-up portrait video, the person smiles and turns their head while wind moves their hair and the camera gently pushes in.",
    "A close-up portrait video, the person blinks and smiles while wind moves their hair and background lights shift softly.",
    "A close-up portrait video, the person turns their head as the camera moves closer and their hair moves continuously in the wind.",
)

DEFAULT_TRAINING_NEGATIVE_PROMPT = (
    "static image, still frame, frozen video, no motion, distorted face, warped eyes, "
    "deformed hair, flickering face, identity shift, bad anatomy, blurry, low quality, "
    "watermark, text artifacts"
)

@dataclass
class PortraitStudentConfig:
    # Sources
    teacher_model_id: str = "Lightricks/LTX-Video"
    teacher_revision: str = "8984fa25007f376c1a299016d0957a37a2f797bb"
    dataset_root: str = "celebv_hq"
    manifest_name: str = "celebv_hq_manifest.csv"
    output_dir: str = "outputs_ltx_student_celebv_hq"
    max_sequence_length: int = 256
    rollout_prompt_bank: Tuple[str, ...] = DEFAULT_ROLLOUT_PROMPT_BANK
    rollout_prompt_bank_probability: float = 0.35
    training_negative_prompt: str = DEFAULT_TRAINING_NEGATIVE_PROMPT

    # Data and the deployment shape
    # LTX requires 8n+1 native frames.
    frames: int = 49
    width: int = 512
    height: int = 512
    fps: int = 25
    train_on_full_dataset: bool = True
    validation_fraction: float = 0.01
    max_train_samples: Optional[int] = None

    # ~983M parameters: fourteen full-width blocks preserve the teacher representation space.
    student_num_layers: int = 14
    student_num_attention_heads: int = 32
    student_attention_head_dim: int = 64
    initialize_from_teacher: bool = True
    gradient_checkpointing: bool = True

    # Optimization
    batch_size: int = 1
    grad_accum_steps: int = 2
    num_workers: int = 2
    epochs: int = 2
    learning_rate: float = 3e-5
    min_learning_rate_ratio: float = 0.10
    warmup_steps: int = 2_000
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0

    # Distillation objective
    teacher_mse_weight: float = 1.0
    teacher_cosine_weight: float = 0.05
    guided_teacher_mse_weight: float = 0.5
    flow_matching_weight: float = 0.25
    x0_reconstruction_weight: float = 0.2
    feature_mse_weight: float = 0.25
    final_feature_mse_weight: float = 0.10
    cfg_pair_probability: float = 0.25
    timestep_high_probability: float = 1 / 3
    timestep_mid_probability: float = 1 / 3
    timestep_low_probability: float = 1 / 3
    face_token_weight: float = 3.0
    background_token_weight: float = 0.35
    temporal_consistency_weight: float = 0.05

    # Train on states produced by the student itself.
    rollout_probability: float = 0.40
    final_rollout_probability: float = 0.50
    rollout_inference_steps: int = 25
    rollout_max_depth: int = 22
    rollout_backprop_steps: int = 3
    rollout_guidance_scale: float = 3.0
    specialization_start_fraction: float = 0.60
    final_guided_teacher_mse_weight: float = 0.75
    final_flow_matching_weight: float = 0.20
    final_x0_reconstruction_weight: float = 0.15

    # IO / validation
    log_every_steps: int = 10
    save_every_steps: int = 1_000
    validate_every_steps: int = 2_000
    # Share of the held-out validation split scored on every validation pass.
    validation_eval_fraction: float = 1.0
    rollout_validation_batches: int = 64
    endpoint_validation_batches: int = 8
    selection_rollout_weight: float = 0.25
    selection_endpoint_weight: float = 0.30
    selection_temporal_weight: float = 0.20
    selection_perceptual_weight: float = 0.20
    selection_motion_weight: float = 0.05
    ema_decay: float = 0.9995
    ema_update_every: int = 10
    ema_start_step: int = 1_000
    seed: int = 42
    dtype: str = "bf16"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def count_parameters_millions(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def face_centric_video_config(cfg: PortraitStudentConfig) -> FaceCentricVideoConfig:
    return FaceCentricVideoConfig(
        width=cfg.width,
        height=cfg.height,
        frames=cfg.frames,
        target_fps=float(cfg.fps),
    )


def _manifest_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class CelebVHQRecord:
    path: str
    clip_id: str
    source_id: str
    caption: str
    regular_eligible: bool


class CelebVHQVideoDataset(Dataset):
    def __init__(self, cfg: PortraitStudentConfig, training: bool, split: str):
        self.cfg = cfg
        self.training = training
        self.epoch = 0
        self.video_config = face_centric_video_config(cfg)

        root = Path(cfg.dataset_root)
        manifest_path = root / cfg.manifest_name
        videos_dir = root / "videos"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing CelebV-HQ manifest: {manifest_path}. "
                "Run download_celebv_hq.py first."
            )

        frame = pd.read_csv(manifest_path)
        required_columns = {
            "video_file",
            "clip_id",
            "ytb_id",
            "caption",
            "regular_eligible",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise RuntimeError(
                f"CelebV-HQ manifest is missing columns: {sorted(missing_columns)}"
            )
        frame["resolved_path"] = frame["video_file"].astype(str).map(
            lambda name: str(videos_dir / name)
        )
        missing_files = frame[
            ~frame["resolved_path"].map(lambda value: Path(value).exists())
        ]
        if not missing_files.empty:
            raise RuntimeError(
                f"CelebV-HQ manifest references {len(missing_files)} missing clips. "
                "Resume download_celebv_hq.py before training."
            )
        frame["caption"] = frame["caption"].astype(str)
        frame["regular_eligible"] = frame["regular_eligible"].map(_manifest_bool)

        if cfg.max_train_samples is not None:
            frame = frame.iloc[: cfg.max_train_samples].copy()
        if len(frame) < 2:
            raise RuntimeError("CelebV-HQ manifest contains fewer than two clips.")

        regular_frame = frame[frame["regular_eligible"]].copy()
        groups = regular_frame["ytb_id"].astype(str).drop_duplicates().tolist()
        if not groups:
            raise RuntimeError("No CelebV-HQ clips are eligible for regular video loss.")
        generator = random.Random(cfg.seed)
        generator.shuffle(groups)
        validation_target = max(
            1, int(round(len(regular_frame) * cfg.validation_fraction))
        )
        group_sizes = (
            regular_frame.assign(ytb_id=regular_frame["ytb_id"].astype(str))[
                "ytb_id"
            ]
            .value_counts()
            .to_dict()
        )
        validation_groups = set()
        validation_count = 0
        for group in groups:
            if validation_count >= validation_target and validation_groups:
                break
            validation_groups.add(group)
            validation_count += int(group_sizes[group])

        if split == "validation":
            selected = regular_frame[
                regular_frame["ytb_id"].astype(str).isin(validation_groups)
            ]
        elif split == "train":
            selected = (
                frame
                if cfg.train_on_full_dataset
                else frame[
                    ~frame["ytb_id"].astype(str).isin(validation_groups)
                ]
            )
        else:
            raise ValueError(f"Unknown split: {split}")
        if selected.empty:
            raise RuntimeError(f"CelebV-HQ {split} split is empty.")

        self.records = [
            CelebVHQRecord(
                path=str(row.resolved_path),
                clip_id=str(row.clip_id),
                source_id=str(row.ytb_id),
                caption=str(row.caption),
                regular_eligible=bool(row.regular_eligible),
            )
            for row in selected.itertuples(index=False)
        ]
        regular_count = sum(record.regular_eligible for record in self.records)
        print(
            f"{split}: {len(self.records)} CelebV-HQ clips "
            f"(regular={regular_count}, rollout-only={len(self.records) - regular_count})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int, salt: str) -> random.Random:
        record = self.records[index]
        digest = hashlib.sha1(
            f"{self.cfg.seed}:{self.epoch}:{record.clip_id}:{salt}".encode("utf-8")
        ).digest()
        return random.Random(
            int.from_bytes(digest[:8], byteorder="little", signed=False)
        )

    def video_item(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        if not record.regular_eligible:
            raise RuntimeError(
                f"Clip {record.clip_id} is rollout-only and cannot provide video loss."
            )
        try:
            window = read_celebv_hq_video_clip(
                record.path,
                self.video_config,
                self.training,
                self._rng(index, "video"),
            )
        except DatasetClipError as error:
            raise RuntimeError(
                f"Failed to decode CelebV-HQ clip {record.clip_id}: {error}"
            ) from error
        return {
            "pixel_values": window.pixel_values,
            "conditioning_image": window.conditioning_image,
            "face_mask": window.face_mask,
            "caption": record.caption,
            "path": record.path,
            "clip_id": record.clip_id,
            "is_rollout": False,
        }

    def rollout_item(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        try:
            image, face_mask = read_celebv_hq_conditioning_frame(
                record.path,
                self.video_config,
                self._rng(index, "rollout"),
            )
        except DatasetClipError as error:
            raise RuntimeError(
                f"Failed to decode CelebV-HQ frame {record.clip_id}: {error}"
            ) from error
        prompt_rng = self._rng(index, "rollout-prompt")
        caption = record.caption
        if self.training and (
            prompt_rng.random() < self.cfg.rollout_prompt_bank_probability
        ):
            caption = prompt_rng.choice(self.cfg.rollout_prompt_bank)
        return {
            "conditioning_image": image,
            "face_mask": face_mask,
            "caption": caption,
            "path": record.path,
            "clip_id": record.clip_id,
            "is_rollout": True,
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.video_item(index)


class CelebVHQTrainingDataset(Dataset):
    """Use every clip; incompatible real motion is routed to teacher rollout."""

    def __init__(self, cfg: PortraitStudentConfig, dataset: CelebVHQVideoDataset):
        self.cfg = cfg
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, request: Tuple[int, float]) -> Dict[str, object]:
        index, rollout_probability = request
        record = self.dataset.records[index]
        digest = hashlib.sha1(
            f"{self.cfg.seed}:{self.dataset.epoch}:{record.clip_id}:batch-kind".encode(
                "utf-8"
            )
        ).digest()
        decision = int.from_bytes(digest[:8], "little") / float(2**64)
        if not record.regular_eligible or decision < rollout_probability:
            return self.dataset.rollout_item(index)
        return self.dataset.video_item(index)


class FixedOrderSampler(Sampler[object]):
    def __init__(self, requests: Sequence[object]):
        self.requests = list(requests)

    def __iter__(self) -> Iterator[object]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


def seed_data_loader_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _student_layer_mapping(student_layers: int, teacher_layers: int) -> List[int]:
    if student_layers == 1:
        return [teacher_layers - 1]
    return [
        int(round(index * (teacher_layers - 1) / (student_layers - 1)))
        for index in range(student_layers)
    ]


@torch.no_grad()
def initialize_student_from_teacher(
    student: LTXVideoTransformer3DModel,
    teacher: LTXVideoTransformer3DModel,
) -> Dict[str, object]:
    teacher_layers = len(teacher.transformer_blocks)
    student_layers = len(student.transformer_blocks)
    layer_mapping = _student_layer_mapping(student_layers, teacher_layers)
    teacher_hidden = int(teacher.config.num_attention_heads * teacher.config.attention_head_dim)
    student_hidden = int(student.config.num_attention_heads * student.config.attention_head_dim)
    if student_hidden != teacher_hidden:
        raise ValueError(
            "Exact teacher initialization requires the same hidden size. "
            "Narrow students need calibrated structural pruning, not tensor slicing."
        )

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    copied_tensors = 0
    copied_values = 0
    total_values = sum(tensor.numel() for tensor in student_state.values())

    for target_name, target_tensor in student_state.items():
        source_name = target_name
        if target_name.startswith("transformer_blocks."):
            parts = target_name.split(".")
            student_index = int(parts[1])
            parts[1] = str(layer_mapping[student_index])
            source_name = ".".join(parts)

        source_tensor = teacher_state.get(source_name)
        if source_tensor is None:
            raise KeyError(f"Teacher tensor is missing: {source_name}")
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shape mismatch for exact initialization: {source_name} "
                f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
            )
        target_tensor.copy_(
            source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
        )
        copied_tensors += 1
        copied_values += target_tensor.numel()

    student.load_state_dict(student_state)
    copied_fraction = copied_values / max(1, total_values)
    if copied_fraction != 1.0:
        raise RuntimeError(f"Incomplete teacher initialization: {copied_fraction:.6f}")
    return {
        "teacher_layers": teacher_layers,
        "student_layers": student_layers,
        "teacher_layer_mapping": layer_mapping,
        "copied_tensors": copied_tensors,
        "copied_value_fraction": copied_fraction,
        "method": "exact copy with evenly spaced full-width teacher blocks",
    }


def build_student(
    teacher: LTXVideoTransformer3DModel,
    cfg: PortraitStudentConfig,
) -> Tuple[LTXVideoTransformer3DModel, Dict[str, object]]:
    student_hidden = cfg.student_num_attention_heads * cfg.student_attention_head_dim
    student_config = dict(teacher.config)
    student_config.update(
        {
            "num_layers": cfg.student_num_layers,
            "num_attention_heads": cfg.student_num_attention_heads,
            "attention_head_dim": cfg.student_attention_head_dim,
            # caption_projection emits inner_dim, so cross-attention must use inner_dim too.
            "cross_attention_dim": student_hidden,
        }
    )
    student = teacher.__class__.from_config(student_config)

    report: Dict[str, object] = {
        "method": "random initialization",
        "teacher_layer_mapping": [],
        "copied_tensors": 0,
        "copied_value_fraction": 0.0,
    }
    if cfg.initialize_from_teacher:
        report = initialize_student_from_teacher(student, teacher)

    print(
        f"Teacher: {count_parameters_millions(teacher):.1f}M, "
        f"layers={teacher.config.num_layers}, hidden="
        f"{teacher.config.num_attention_heads * teacher.config.attention_head_dim}"
    )
    print(
        f"Student: {count_parameters_millions(student):.1f}M, "
        f"layers={student.config.num_layers}, hidden={student_hidden}"
    )
    print("Initialization:", report)
    return student, report


def calculate_ltx_shift(
    sequence_length: int,
    base_sequence_length: int = 256,
    max_sequence_length: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_sequence_length - base_sequence_length)
    intercept = base_shift - slope * base_sequence_length
    return sequence_length * slope + intercept


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    generated_token_mask: torch.Tensor,
) -> torch.Tensor:
    mask = generated_token_mask.unsqueeze(-1).to(prediction.dtype)
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).square() * mask).sum() / denominator


def masked_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    generated_token_mask: torch.Tensor,
) -> torch.Tensor:
    distance = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    mask = generated_token_mask.to(distance.dtype)
    return (distance * mask).sum() / mask.sum().clamp_min(1.0)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    generated_token_mask: torch.Tensor,
) -> torch.Tensor:
    mask = generated_token_mask.unsqueeze(-1).to(prediction.dtype)
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    return (elementwise * mask).sum() / denominator


def latent_temporal_distance(
    predicted_x0: torch.Tensor,
    latent_face_mask: torch.Tensor,
    target_x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    predicted_delta = predicted_x0[:, :, 1:] - predicted_x0[:, :, :-1]
    mask = latent_face_mask.expand(
        -1,
        1,
        predicted_delta.shape[2],
        -1,
        -1,
    )
    if target_x0 is not None:
        target_delta = target_x0[:, :, 1:] - target_x0[:, :, :-1]
        elementwise = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
            reduction="none",
        )
    elif predicted_delta.shape[2] > 1:
        acceleration = predicted_delta[:, :, 1:] - predicted_delta[:, :, :-1]
        mask = mask[:, :, 1:]
        elementwise = acceleration.abs()
    else:
        return predicted_delta.new_zeros(())
    denominator = mask.sum().clamp_min(1.0) * predicted_delta.shape[1]
    return (elementwise * mask).sum() / denominator


def latent_motion_score(
    latents: torch.Tensor,
    latent_face_mask: torch.Tensor,
) -> torch.Tensor:
    delta = (latents[:, :, 1:] - latents[:, :, :-1]).abs()
    mask = latent_face_mask.expand(-1, 1, delta.shape[2], -1, -1)
    denominator = mask.sum().clamp_min(1.0) * delta.shape[1]
    return (delta * mask).sum() / denominator


class CPUModelEMA:
    """Low-frequency BF16 EMA that does not consume training VRAM."""

    def __init__(self, model: nn.Module, decay: float, update_every: int):
        self.decay = float(decay)
        self.update_every = max(1, int(update_every))
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module, global_step: int) -> None:
        if global_step % self.update_every != 0:
            return
        self.num_updates += 1
        warm_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        update_weight = 1.0 - warm_decay
        for name, parameter in model.named_parameters():
            source = parameter.detach().to(device="cpu", dtype=torch.bfloat16)
            self.shadow[name].lerp_(source, update_weight)

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.update_every = int(state.get("update_every", self.update_every))
        self.num_updates = int(state.get("num_updates", 0))
        saved_shadow = state.get("shadow")
        if isinstance(saved_shadow, dict):
            missing = set(self.shadow).difference(saved_shadow)
            if missing:
                raise RuntimeError(f"EMA state is missing parameters: {sorted(missing)[:5]}")
            self.shadow = {
                name: tensor.detach().to(device="cpu", dtype=torch.bfloat16)
                for name, tensor in saved_shadow.items()
            }

    def model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if name in self.shadow:
                result[name] = self.shadow[name]
            else:
                result[name] = tensor.detach().to("cpu")
        return result


class PortraitStudentTrainer:
    def __init__(self, cfg: PortraitStudentConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("Training the LTX student requires a CUDA GPU.")
        if cfg.batch_size != 1:
            raise ValueError(
                "CelebVHQTrainingDataset requires batch_size=1 because rollout and "
                "regular video samples have different schemas."
            )
        timestep_probability_sum = (
            cfg.timestep_high_probability
            + cfg.timestep_mid_probability
            + cfg.timestep_low_probability
        )
        if not math.isclose(timestep_probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "Timestep high/mid/low probabilities must sum to 1, got "
                f"{timestep_probability_sum:.6f}"
            )
        if (
            cfg.rollout_max_depth + cfg.rollout_backprop_steps
            > cfg.rollout_inference_steps
        ):
            raise ValueError(
                "rollout_max_depth + rollout_backprop_steps must not exceed "
                "rollout_inference_steps."
            )

        self.cfg = cfg
        self.device = torch.device("cuda")
        self.teacher_dtype = torch_dtype_from_name(cfg.dtype)
        seed_everything(cfg.seed)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = Path(cfg.dataset_root) / cfg.manifest_name
        signature_config = asdict(cfg)
        for operational_key in (
            "log_every_steps",
            "save_every_steps",
            "validate_every_steps",
            "validation_eval_fraction",
        ):
            signature_config.pop(operational_key, None)
        signature_payload = {
            "config": signature_config,
            "manifest_sha256": file_sha256(manifest_path),
        }
        self.run_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.train_video_dataset = CelebVHQVideoDataset(
            cfg, training=True, split="train"
        )
        self.train_dataset = CelebVHQTrainingDataset(
            cfg, self.train_video_dataset
        )
        self.validation_dataset = CelebVHQVideoDataset(
            cfg, training=False, split="validation"
        )

        print("Loading teacher pipeline with T5 to encode dataset captions...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            cfg.teacher_model_id,
            revision=cfg.teacher_revision,
            torch_dtype=self.teacher_dtype,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        # The published LTX tokenizer still declares model_max_length=128, but the
        # video model was trained with prompts up to 256 tokens.
        if self.pipe.tokenizer is not None:
            self.pipe.tokenizer.model_max_length = cfg.max_sequence_length

        self.teacher = self.pipe.transformer.eval()
        self.teacher.requires_grad_(False)
        self.pipe.vae.eval()
        self.pipe.vae.requires_grad_(False)
        self.pipe.vae.enable_tiling()
        self.perceptual_metric = lpips.LPIPS(net="alex", verbose=False).eval()
        self.perceptual_metric.requires_grad_(False)
        temporal_ratio = int(self.pipe.vae_temporal_compression_ratio)
        if (cfg.frames - 1) % temporal_ratio != 0:
            raise ValueError(
                f"LTX native frame count must satisfy frames=8n+1; got {cfg.frames}"
            )

        self.student, self.initialization_report = build_student(self.teacher, cfg)
        self.student = self.student.to(device=self.device, dtype=torch.float32)
        if cfg.gradient_checkpointing:
            self.student.enable_gradient_checkpointing()
        self.student.train()
        self.ema = CPUModelEMA(
            self.student,
            decay=cfg.ema_decay,
            update_every=cfg.ema_update_every,
        )

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.teacher_dtype == torch.float16
        )

        updates_per_epoch = math.ceil(
            len(self.train_dataset) / (cfg.batch_size * cfg.grad_accum_steps)
        )
        self.total_optimizer_steps = updates_per_epoch * cfg.epochs
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._learning_rate_multiplier
        )

        self.rollout_scheduler = copy.deepcopy(self.pipe.scheduler)
        self.latent_frames = (cfg.frames - 1) // self.pipe.vae_temporal_compression_ratio + 1
        self.latent_height = cfg.height // self.pipe.vae_spatial_compression_ratio
        self.latent_width = cfg.width // self.pipe.vae_spatial_compression_ratio
        self.sequence_length = self.latent_frames * self.latent_height * self.latent_width
        scheduler_config = self.pipe.scheduler.config
        self.scheduler_mu = calculate_ltx_shift(
            self.sequence_length,
            base_sequence_length=int(
                scheduler_config.get("base_image_seq_len", 256)
            ),
            max_sequence_length=int(
                scheduler_config.get("max_image_seq_len", 4096)
            ),
            base_shift=float(scheduler_config.get("base_shift", 0.5)),
            max_shift=float(scheduler_config.get("max_shift", 1.15)),
        )
        self.rope_interpolation_scale = (
            self.pipe.vae_temporal_compression_ratio / cfg.fps,
            self.pipe.vae_spatial_compression_ratio,
            self.pipe.vae_spatial_compression_ratio,
        )

        self.negative_prompt_embeds, self.negative_prompt_mask = self._encode_captions(
            [cfg.training_negative_prompt]
        )

        self.global_step = 0
        self.videos_seen = 0
        self.start_epoch = 0
        self.start_sample_in_epoch = 0
        self.best_validation_loss = float("inf")
        self.optimizer.zero_grad(set_to_none=True)
        self._resume_if_available()

    def _learning_rate_multiplier(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return (step + 1) / max(1, self.cfg.warmup_steps)
        progress = (step - self.cfg.warmup_steps) / max(
            1, self.total_optimizer_steps - self.cfg.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = self.cfg.min_learning_rate_ratio
        return floor + (1.0 - floor) * cosine

    def _training_progress(self) -> float:
        return min(1.0, max(0.0, self.global_step / max(1, self.total_optimizer_steps)))

    def _specialization_mix(self) -> float:
        return self._specialization_mix_at(self._training_progress())

    def _specialization_mix_at(self, progress: float) -> float:
        start = float(self.cfg.specialization_start_fraction)
        if progress <= start:
            return 0.0
        return min(1.0, (progress - start) / max(1e-6, 1.0 - start))

    def _curriculum_value(self, initial: float, final: float) -> float:
        mix = self._specialization_mix()
        return initial + mix * (final - initial)

    def _rollout_probability_at(self, progress: float) -> float:
        mix = self._specialization_mix_at(progress)
        return (
            self.cfg.rollout_probability
            + mix
            * (self.cfg.final_rollout_probability - self.cfg.rollout_probability)
        )

    def _make_train_loader(self, epoch: int, start_sample: int) -> DataLoader:
        self.train_video_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(self.cfg.seed + epoch)
        order = torch.randperm(len(self.train_dataset), generator=generator).tolist()
        total_samples = len(self.train_dataset) * self.cfg.epochs
        requests = [
            (
                index,
                self._rollout_probability_at(
                    (epoch * len(self.train_dataset) + position)
                    / max(1, total_samples - 1)
                ),
            )
            for position, index in enumerate(order)
        ]
        requests = requests[start_sample:]
        sampler = FixedOrderSampler(requests)
        if requests:
            print(
                f"epoch={epoch} rollout_probability="
                f"{requests[0][1]:.3f}->{requests[-1][1]:.3f}"
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.cfg.num_workers > 0,
            worker_init_fn=partial(
                seed_data_loader_worker,
                base_seed=self.cfg.seed + epoch * 10_000,
            ),
        )

    def _make_validation_loader(self) -> DataLoader:
        self.validation_dataset.set_epoch(0)
        return DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=min(1, self.cfg.num_workers),
            pin_memory=True,
            drop_last=False,
        )

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.pipe._normalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )

    @torch.no_grad()
    def _encode_conditioning_batch(
        self, batch: Dict[str, object]
    ) -> torch.Tensor:
        image = batch["conditioning_image"]
        assert isinstance(image, torch.Tensor)
        image = image.unsqueeze(2).to(
            self.device, self.teacher_dtype, non_blocking=True
        )
        with torch.autocast("cuda", dtype=self.teacher_dtype):
            image_distribution = self.pipe.vae.encode(image).latent_dist
            image_latents = image_distribution.sample()
        image_latents = self._normalize_latents(image_latents).float()
        return image_latents[:, :, :1]

    @torch.no_grad()
    def _encode_video_batch(
        self, batch: Dict[str, object]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video = batch["pixel_values"]
        assert isinstance(video, torch.Tensor)
        video = video.permute(0, 2, 1, 3, 4).contiguous().to(
            self.device, self.teacher_dtype, non_blocking=True
        )
        with torch.autocast("cuda", dtype=self.teacher_dtype):
            video_distribution = self.pipe.vae.encode(video).latent_dist
            clean_latents = video_distribution.sample()
        clean_latents = self._normalize_latents(clean_latents).float()
        image_latents = self._encode_conditioning_batch(batch)
        return clean_latents, image_latents

    def _pack(self, latents: torch.Tensor) -> torch.Tensor:
        return self.pipe._pack_latents(
            latents,
            self.student.config.patch_size,
            self.student.config.patch_size_t,
        )

    def _unpack(self, latents: torch.Tensor) -> torch.Tensor:
        return self.pipe._unpack_latents(
            latents,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.student.config.patch_size,
            self.student.config.patch_size_t,
        )

    def _conditioning_mask(
        self, batch_size: int, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_5d = torch.zeros(
            batch_size,
            1,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=dtype,
        )
        mask_5d[:, :, 0] = 1.0
        packed = self._pack(mask_5d).squeeze(-1)
        generated = 1.0 - packed
        return mask_5d, generated

    def _latent_face_mask(self, batch: Dict[str, object]) -> torch.Tensor:
        face_mask = batch["face_mask"]
        assert isinstance(face_mask, torch.Tensor)
        if face_mask.ndim == 3:
            face_mask = face_mask.unsqueeze(1)
        face_mask = face_mask.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        resized = F.interpolate(
            face_mask,
            size=(self.latent_height, self.latent_width),
            mode="area",
        )
        return resized.unsqueeze(2)

    def _face_token_weights(
        self,
        batch: Dict[str, object],
        generated_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_face_mask = self._latent_face_mask(batch)
        expanded = latent_face_mask.expand(
            -1,
            -1,
            self.latent_frames,
            -1,
            -1,
        )
        packed_face = self._pack(expanded).squeeze(-1).clamp(0.0, 1.0)
        spatial_weight = (
            self.cfg.background_token_weight
            + (self.cfg.face_token_weight - self.cfg.background_token_weight)
            * packed_face
        )
        return generated_token_mask * spatial_weight, latent_face_mask

    @torch.no_grad()
    def _encode_captions(
        self, captions: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pipe.text_encoder is None or self.pipe.tokenizer is None:
            raise RuntimeError("Dataset captions require the LTX T5 text encoder.")
        embeds, mask, _, _ = self.pipe.encode_prompt(
            prompt=list(captions),
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=self.cfg.max_sequence_length,
            device=self.device,
            dtype=self.teacher_dtype,
        )
        return embeds, mask

    def _encode_batch_captions(
        self, batch: Dict[str, object]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_captions = batch["caption"]
        if isinstance(raw_captions, str):
            captions = [raw_captions]
        else:
            captions = [str(caption) for caption in raw_captions]
        return self._encode_captions(captions)

    def _token_timesteps(
        self, timesteps: torch.Tensor, generated_token_mask: torch.Tensor
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(generated_token_mask.shape[0])
        return timesteps.unsqueeze(-1) * generated_token_mask

    def _model_forward(
        self,
        model: LTXVideoTransformer3DModel,
        packed_latents: torch.Tensor,
        token_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = model(
            hidden_states=packed_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=token_timesteps,
            encoder_attention_mask=prompt_mask,
            num_frames=self.latent_frames,
            height=self.latent_height,
            width=self.latent_width,
            rope_interpolation_scale=self.rope_interpolation_scale,
            return_dict=False,
        )[0]
        return output.float()

    def _cfg_model_forward(
        self,
        model: LTXVideoTransformer3DModel,
        packed_latents: torch.Tensor,
        token_timesteps: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        positive_prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = packed_latents.shape[0]
        model_latents = torch.cat([packed_latents, packed_latents], dim=0)
        model_timesteps = torch.cat([token_timesteps, token_timesteps], dim=0)
        prompt_embeds = torch.cat(
            [
                self.negative_prompt_embeds.expand(batch_size, -1, -1),
                positive_prompt_embeds,
            ],
            dim=0,
        )
        prompt_mask = torch.cat(
            [
                self.negative_prompt_mask.expand(batch_size, -1),
                positive_prompt_mask,
            ],
            dim=0,
        )
        negative_prediction, positive_prediction = self._model_forward(
            model,
            model_latents,
            model_timesteps,
            prompt_embeds,
            prompt_mask,
        ).chunk(2)
        scale = self.cfg.rollout_guidance_scale
        guided_prediction = negative_prediction + scale * (
            positive_prediction - negative_prediction
        )
        return guided_prediction, negative_prediction, positive_prediction

    def _guided_model_forward(
        self,
        model: LTXVideoTransformer3DModel,
        packed_latents: torch.Tensor,
        token_timesteps: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        positive_prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        guided_prediction, _, _ = self._cfg_model_forward(
            model,
            packed_latents,
            token_timesteps,
            positive_prompt_embeds,
            positive_prompt_mask,
        )
        return guided_prediction

    def _model_forward_with_features(
        self,
        model: LTXVideoTransformer3DModel,
        block_indices: Sequence[int],
        packed_latents: torch.Tensor,
        token_timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        for block_index in block_indices:
            def capture_output(_module, _inputs, output, index=block_index):
                captured[index] = output

            handles.append(
                model.transformer_blocks[block_index].register_forward_hook(
                    capture_output
                )
            )

        try:
            prediction = self._model_forward(
                model,
                packed_latents,
                token_timesteps,
                prompt_embeds,
                prompt_mask,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [index for index in block_indices if index not in captured]
        if missing:
            raise RuntimeError(f"Feature hooks did not run for blocks: {missing}")
        return prediction, [captured[index] for index in block_indices]

    def _distillation_losses(
        self,
        student_prediction: torch.Tensor,
        teacher_prediction: torch.Tensor,
        generated_token_mask: torch.Tensor,
        flow_target: Optional[torch.Tensor],
        flow_prediction: Optional[torch.Tensor] = None,
        flow_token_mask: Optional[torch.Tensor] = None,
        feature_loss: Optional[torch.Tensor] = None,
        feature_loss_weight: float = 0.0,
        x0_loss: Optional[torch.Tensor] = None,
        temporal_loss: Optional[torch.Tensor] = None,
        guided_student_prediction: Optional[torch.Tensor] = None,
        guided_teacher_prediction: Optional[torch.Tensor] = None,
        guided_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        teacher_mse = masked_mse(
            student_prediction, teacher_prediction, generated_token_mask
        )
        teacher_cosine = masked_cosine_distance(
            student_prediction, teacher_prediction, generated_token_mask
        )
        if flow_target is None:
            flow_loss = teacher_mse.new_zeros(())
        else:
            effective_flow_mask = (
                generated_token_mask
                if flow_token_mask is None
                else flow_token_mask
            )
            effective_flow_prediction = (
                student_prediction
                if flow_prediction is None
                else flow_prediction
            )
            flow_loss = masked_mse(
                effective_flow_prediction, flow_target, effective_flow_mask
            )
        if feature_loss is None:
            feature_loss = teacher_mse.new_zeros(())
        if x0_loss is None:
            x0_loss = teacher_mse.new_zeros(())
        if temporal_loss is None:
            temporal_loss = teacher_mse.new_zeros(())
        if (
            guided_student_prediction is None
            or guided_teacher_prediction is None
        ):
            guided_mse = teacher_mse.new_zeros(())
        else:
            effective_guided_mask = (
                generated_token_mask
                if guided_token_mask is None
                else guided_token_mask
            )
            guided_mse = masked_mse(
                guided_student_prediction,
                guided_teacher_prediction,
                effective_guided_mask,
            )

        guided_weight = self._curriculum_value(
            self.cfg.guided_teacher_mse_weight,
            self.cfg.final_guided_teacher_mse_weight,
        )
        flow_weight = self._curriculum_value(
            self.cfg.flow_matching_weight,
            self.cfg.final_flow_matching_weight,
        )
        x0_weight = self._curriculum_value(
            self.cfg.x0_reconstruction_weight,
            self.cfg.final_x0_reconstruction_weight,
        )
        total = (
            self.cfg.teacher_mse_weight * teacher_mse
            + self.cfg.teacher_cosine_weight * teacher_cosine
            + guided_weight * guided_mse
            + flow_weight * flow_loss
            + x0_weight * x0_loss
            + feature_loss_weight * feature_loss
            + self.cfg.temporal_consistency_weight * temporal_loss
        )
        return {
            "loss": total,
            "teacher_mse": teacher_mse,
            "teacher_cosine": teacher_cosine,
            "guided_mse": guided_mse,
            "flow_mse": flow_loss,
            "x0_smooth_l1": x0_loss,
            "feature_distance": feature_loss,
            "temporal_distance": temporal_loss,
        }

    def _feature_distillation_weight(self) -> float:
        return self._curriculum_value(
            self.cfg.feature_mse_weight,
            self.cfg.final_feature_mse_weight,
        )

    def _sample_training_timestep(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._set_rollout_timesteps()
        timesteps = torch.as_tensor(
            self.rollout_scheduler.timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        sigmas = torch.as_tensor(
            self.rollout_scheduler.sigmas[:-1],
            device=self.device,
            dtype=torch.float32,
        )
        schedule_length = min(len(timesteps), len(sigmas))
        probabilities = torch.tensor(
            [
                self.cfg.timestep_high_probability,
                self.cfg.timestep_mid_probability,
                self.cfg.timestep_low_probability,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        categories = torch.multinomial(
            probabilities,
            num_samples=batch_size,
            replacement=True,
        )
        boundaries = [0, schedule_length // 3, (2 * schedule_length) // 3, schedule_length]
        sampled_indices = []
        for category in categories.tolist():
            start = boundaries[category]
            stop = max(start + 1, boundaries[category + 1])
            sampled_indices.append(
                torch.randint(start, stop, (), device=self.device)
            )
        indices = torch.stack(sampled_indices)
        return sigmas[indices], timesteps[indices]

    def _regular_distillation_batch(
        self,
        clean_latents: torch.Tensor,
        image_latents: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        positive_prompt_mask: torch.Tensor,
        batch: Dict[str, object],
        allow_negative: bool,
    ) -> Dict[str, torch.Tensor]:
        batch_size = clean_latents.shape[0]
        noise = torch.randn_like(clean_latents)
        sigma, timesteps = self._sample_training_timestep(batch_size)
        sigma_5d = sigma.view(batch_size, 1, 1, 1, 1)
        noisy_latents = (1.0 - sigma_5d) * clean_latents + sigma_5d * noise
        noisy_latents[:, :, :1] = image_latents

        packed_latents = self._pack(noisy_latents)
        _, generated_token_mask = self._conditioning_mask(batch_size)
        token_weights, latent_face_mask = self._face_token_weights(
            batch, generated_token_mask
        )
        token_timesteps = self._token_timesteps(timesteps, generated_token_mask)
        use_cfg_pair = allow_negative and (
            torch.rand((), device=self.device).item() < self.cfg.cfg_pair_probability
        )
        if use_cfg_pair:
            model_latents = torch.cat([packed_latents, packed_latents], dim=0)
            model_timesteps = torch.cat([token_timesteps, token_timesteps], dim=0)
            prompt_embeds = torch.cat(
                [
                    self.negative_prompt_embeds.expand(batch_size, -1, -1),
                    positive_prompt_embeds,
                ],
                dim=0,
            )
            prompt_mask = torch.cat(
                [
                    self.negative_prompt_mask.expand(batch_size, -1),
                    positive_prompt_mask,
                ],
                dim=0,
            )
            teacher_token_mask = torch.cat(
                [token_weights, token_weights], dim=0
            )
        else:
            model_latents = packed_latents
            model_timesteps = token_timesteps
            prompt_embeds = positive_prompt_embeds
            prompt_mask = positive_prompt_mask
            teacher_token_mask = token_weights

        feature_loss_weight = self._feature_distillation_weight()
        use_feature_distillation = feature_loss_weight > 0.0
        if use_feature_distillation:
            teacher_block_indices = list(
                self.initialization_report["teacher_layer_mapping"]
            )
            student_block_indices = list(range(len(self.student.transformer_blocks)))
            with torch.no_grad(), torch.autocast("cuda", dtype=self.teacher_dtype):
                teacher_prediction, teacher_features = (
                    self._model_forward_with_features(
                        self.teacher,
                        teacher_block_indices,
                        model_latents,
                        model_timesteps,
                        prompt_embeds,
                        prompt_mask,
                    )
                )
            with torch.autocast("cuda", dtype=self.teacher_dtype):
                student_prediction, student_features = (
                    self._model_forward_with_features(
                        self.student,
                        student_block_indices,
                        model_latents,
                        model_timesteps,
                        prompt_embeds,
                        prompt_mask,
                    )
                )
            feature_loss = torch.stack(
                [
                    masked_cosine_distance(
                        student_feature.float(),
                        teacher_feature.float(),
                        teacher_token_mask,
                    )
                    for student_feature, teacher_feature in zip(
                        student_features, teacher_features
                    )
                ]
            ).mean()
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=self.teacher_dtype):
                teacher_prediction = self._model_forward(
                    self.teacher,
                    model_latents,
                    model_timesteps,
                    prompt_embeds,
                    prompt_mask,
                )
            with torch.autocast("cuda", dtype=self.teacher_dtype):
                student_prediction = self._model_forward(
                    self.student,
                    model_latents,
                    model_timesteps,
                    prompt_embeds,
                    prompt_mask,
                )
            feature_loss = None

        if use_cfg_pair:
            student_negative, student_positive = student_prediction.chunk(2)
            teacher_negative, teacher_positive = teacher_prediction.chunk(2)
            scale = self.cfg.rollout_guidance_scale
            guided_student_prediction = student_negative + scale * (
                student_positive - student_negative
            )
            guided_teacher_prediction = teacher_negative + scale * (
                teacher_positive - teacher_negative
            )
            flow_prediction = student_positive
        else:
            guided_student_prediction = None
            guided_teacher_prediction = None
            flow_prediction = student_prediction

        flow_target = self._pack(noise - clean_latents)
        packed_clean_latents = self._pack(clean_latents)
        x0_prediction = (
            guided_student_prediction
            if guided_student_prediction is not None
            else flow_prediction
        )
        predicted_x0 = packed_latents - sigma[:, None, None] * x0_prediction
        x0_loss = masked_smooth_l1(
            predicted_x0,
            packed_clean_latents,
            token_weights,
        )
        unpacked_x0 = self._unpack(predicted_x0)
        predicted_x0_5d = torch.cat(
            [image_latents, unpacked_x0[:, :, 1:]],
            dim=2,
        )
        temporal_target = torch.cat(
            [image_latents, clean_latents[:, :, 1:]],
            dim=2,
        )
        temporal_loss = latent_temporal_distance(
            predicted_x0_5d,
            latent_face_mask,
            target_x0=temporal_target,
        )
        return self._distillation_losses(
            student_prediction,
            teacher_prediction,
            teacher_token_mask,
            flow_target,
            flow_prediction=flow_prediction,
            flow_token_mask=token_weights,
            feature_loss=feature_loss,
            feature_loss_weight=feature_loss_weight,
            x0_loss=x0_loss,
            temporal_loss=temporal_loss,
            guided_student_prediction=guided_student_prediction,
            guided_teacher_prediction=guided_teacher_prediction,
            guided_token_mask=token_weights,
        )

    def _set_rollout_timesteps(self) -> None:
        steps = self.cfg.rollout_inference_steps
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        self.rollout_scheduler.set_timesteps(
            sigmas=sigmas,
            device=self.device,
            mu=self.scheduler_mu,
        )

    def _current_rollout_max_depth(self) -> int:
        maximum = self.cfg.rollout_max_depth
        progress = self.global_step / max(1, self.total_optimizer_steps)
        if progress < 0.10:
            return min(3, maximum)
        if progress < 0.30:
            return min(8, maximum)
        if progress < 0.60:
            return min(16, maximum)
        return maximum

    def _rollout_step(
        self,
        packed_state: torch.Tensor,
        prediction: torch.Tensor,
        step_index: int,
        generated_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = self.rollout_scheduler.sigmas
        delta_sigma = torch.as_tensor(
            sigmas[step_index + 1] - sigmas[step_index],
            device=self.device,
            dtype=torch.float32,
        )
        return (
            packed_state
            + delta_sigma * prediction * generated_token_mask.unsqueeze(-1)
        )

    def _rollout_distillation_batch(
        self,
        image_latents: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        positive_prompt_mask: torch.Tensor,
        batch: Dict[str, object],
        forced_depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = image_latents.shape[0]
        self._set_rollout_timesteps()
        _, generated_token_mask = self._conditioning_mask(batch_size)
        token_weights, latent_face_mask = self._face_token_weights(
            batch, generated_token_mask
        )

        latent_state = torch.randn(
            batch_size,
            image_latents.shape[1],
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=torch.float32,
        )
        latent_state[:, :, :1] = image_latents
        schedule_max_depth = (
            len(self.rollout_scheduler.timesteps) - self.cfg.rollout_backprop_steps
        )
        if forced_depth is None:
            maximum_depth = min(
                self._current_rollout_max_depth(),
                schedule_max_depth,
            )
            depth = random.randint(0, maximum_depth)
        else:
            depth = min(
                max(0, int(forced_depth)),
                self.cfg.rollout_max_depth,
                schedule_max_depth,
            )

        packed_state = self._pack(latent_state)
        was_training = self.student.training
        self.student.eval()
        with torch.no_grad():
            for step_index, timestep in enumerate(
                self.rollout_scheduler.timesteps[:depth]
            ):
                token_timesteps = self._token_timesteps(
                    timestep, generated_token_mask
                )
                with torch.autocast("cuda", dtype=self.teacher_dtype):
                    prediction = self._guided_model_forward(
                        self.student,
                        packed_state,
                        token_timesteps,
                        positive_prompt_embeds,
                        positive_prompt_mask,
                    )
                packed_state = self._rollout_step(
                    packed_state,
                    prediction,
                    step_index,
                    generated_token_mask,
                )
        self.student.train(was_training)

        student_state = packed_state
        teacher_token_mask = torch.cat([token_weights, token_weights], dim=0)
        totals: Dict[str, torch.Tensor] = {}

        for step_index in range(depth, depth + self.cfg.rollout_backprop_steps):
            timestep = self.rollout_scheduler.timesteps[step_index]
            token_timesteps = self._token_timesteps(
                timestep, generated_token_mask
            )
            teacher_input = student_state.detach()
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=self.teacher_dtype
            ):
                teacher_guided, teacher_negative, teacher_positive = (
                    self._cfg_model_forward(
                        self.teacher,
                        teacher_input,
                        token_timesteps,
                        positive_prompt_embeds,
                        positive_prompt_mask,
                    )
                )
            with torch.autocast("cuda", dtype=self.teacher_dtype):
                student_guided, student_negative, student_positive = (
                    self._cfg_model_forward(
                        self.student,
                        student_state,
                        token_timesteps,
                        positive_prompt_embeds,
                        positive_prompt_mask,
                    )
                )

            sigma = torch.as_tensor(
                self.rollout_scheduler.sigmas[step_index],
                device=self.device,
                dtype=torch.float32,
            )
            student_x0 = student_state - sigma * student_guided
            teacher_x0 = teacher_input - sigma * teacher_guided
            x0_loss = masked_smooth_l1(
                student_x0,
                teacher_x0,
                token_weights,
            )
            student_x0_5d = self._unpack(student_x0)
            teacher_x0_5d = self._unpack(teacher_x0)
            student_x0_5d = torch.cat(
                [image_latents, student_x0_5d[:, :, 1:]],
                dim=2,
            )
            teacher_x0_5d = torch.cat(
                [image_latents, teacher_x0_5d[:, :, 1:]],
                dim=2,
            )
            temporal_loss = latent_temporal_distance(
                student_x0_5d,
                latent_face_mask,
                target_x0=teacher_x0_5d,
            )

            step_losses = self._distillation_losses(
                torch.cat([student_negative, student_positive], dim=0),
                torch.cat([teacher_negative, teacher_positive], dim=0),
                teacher_token_mask,
                flow_target=None,
                x0_loss=x0_loss,
                temporal_loss=temporal_loss,
                guided_student_prediction=student_guided,
                guided_teacher_prediction=teacher_guided,
                guided_token_mask=token_weights,
            )
            for name, value in step_losses.items():
                totals[name] = totals.get(name, value.new_zeros(())) + value

            student_state = self._rollout_step(
                student_state,
                student_guided,
                step_index,
                generated_token_mask,
            )

        losses = {
            name: value / self.cfg.rollout_backprop_steps
            for name, value in totals.items()
        }
        losses["rollout_depth"] = losses["loss"].new_tensor(float(depth))
        losses["rollout_steps"] = losses["loss"].new_tensor(
            float(self.cfg.rollout_backprop_steps)
        )
        return losses

    def _compute_batch_losses(
        self,
        batch: Dict[str, object],
        use_rollout: bool,
        allow_negative: bool,
        forced_rollout_depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_rollout:
            clean_latents = None
            image_latents = self._encode_conditioning_batch(batch)
        else:
            clean_latents, image_latents = self._encode_video_batch(batch)
        positive_prompt_embeds, positive_prompt_mask = self._encode_batch_captions(
            batch
        )
        if use_rollout:
            return self._rollout_distillation_batch(
                image_latents,
                positive_prompt_embeds,
                positive_prompt_mask,
                batch,
                forced_depth=forced_rollout_depth,
            )
        assert clean_latents is not None
        return self._regular_distillation_batch(
            clean_latents,
            image_latents,
            positive_prompt_embeds,
            positive_prompt_mask,
            batch,
            allow_negative=allow_negative,
        )

    @torch.no_grad()
    def _full_rollout_endpoint(
        self,
        model: LTXVideoTransformer3DModel,
        image_latents: torch.Tensor,
        initial_noise: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        positive_prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._set_rollout_timesteps()
        batch_size = image_latents.shape[0]
        _, generated_token_mask = self._conditioning_mask(batch_size)
        latent_state = initial_noise.clone()
        latent_state[:, :, :1] = image_latents
        for timestep in self.rollout_scheduler.timesteps:
            packed = self._pack(latent_state)
            token_timesteps = self._token_timesteps(
                timestep, generated_token_mask
            )
            with torch.autocast("cuda", dtype=self.teacher_dtype):
                prediction = self._guided_model_forward(
                    model,
                    packed,
                    token_timesteps,
                    positive_prompt_embeds,
                    positive_prompt_mask,
                )
            prediction_5d = self._unpack(prediction)
            next_generated = self.rollout_scheduler.step(
                prediction_5d[:, :, 1:],
                timestep,
                latent_state[:, :, 1:],
                return_dict=False,
            )[0]
            latent_state = torch.cat(
                [latent_state[:, :, :1], next_generated],
                dim=2,
            )
        return latent_state

    @torch.no_grad()
    def _decode_endpoint_frames(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self.pipe._denormalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        ).to(self.teacher_dtype)
        timestep = None
        if self.pipe.vae.config.timestep_conditioning:
            timestep = torch.zeros(
                latents.shape[0],
                device=self.device,
                dtype=self.teacher_dtype,
            )
        video = self.pipe.vae.decode(
            latents,
            timestep,
            return_dict=False,
        )[0]
        frame_indices = [video.shape[2] // 2, video.shape[2] - 1]
        frames = video[:, :, frame_indices].permute(0, 2, 1, 3, 4)
        frames = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
        frames = F.interpolate(
            frames.float(),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        return frames.clamp(-1.0, 1.0).cpu()

    @torch.no_grad()
    def _endpoint_validation_batch(
        self,
        batch: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        image_latents = self._encode_conditioning_batch(batch)
        batch_size = image_latents.shape[0]
        prompt_embeds, prompt_mask = self._encode_batch_captions(batch)
        initial_noise = torch.randn(
            batch_size,
            image_latents.shape[1],
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=torch.float32,
        )
        initial_noise[:, :, :1] = image_latents

        student_endpoint = self._full_rollout_endpoint(
            self.student,
            image_latents,
            initial_noise,
            prompt_embeds,
            prompt_mask,
        )
        teacher_endpoint = self._full_rollout_endpoint(
            self.teacher,
            image_latents,
            initial_noise,
            prompt_embeds,
            prompt_mask,
        )

        _, generated_token_mask = self._conditioning_mask(batch_size)
        token_weights, latent_face_mask = self._face_token_weights(
            batch, generated_token_mask
        )
        packed_student = self._pack(student_endpoint)
        packed_teacher = self._pack(teacher_endpoint)
        endpoint_mse = masked_mse(
            packed_student,
            packed_teacher,
            token_weights,
        )
        endpoint_cosine = masked_cosine_distance(
            packed_student,
            packed_teacher,
            token_weights,
        )
        student_motion = latent_motion_score(
            student_endpoint,
            latent_face_mask,
        )
        teacher_motion = latent_motion_score(
            teacher_endpoint,
            latent_face_mask,
        )
        motion_error = (student_motion - teacher_motion).abs()
        temporal_distance = latent_temporal_distance(
            student_endpoint,
            latent_face_mask,
            target_x0=teacher_endpoint,
        )
        student_frames = self._decode_endpoint_frames(student_endpoint)
        teacher_frames = self._decode_endpoint_frames(teacher_endpoint)
        perceptual_distance = self.perceptual_metric(
            student_frames,
            teacher_frames,
        ).mean()
        return {
            "mse": endpoint_mse,
            "cosine": endpoint_cosine,
            "motion": student_motion,
            "teacher_motion": teacher_motion,
            "motion_error": motion_error,
            "temporal_distance": temporal_distance,
            "lpips": perceptual_distance,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        was_training = self.student.training
        seed_everything(self.cfg.seed + 1_000_000)

        eval_fraction = float(self.cfg.validation_eval_fraction)
        if not 0.0 < eval_fraction <= 1.0:
            raise ValueError(
                "validation_eval_fraction must be in (0, 1], "
                f"got {eval_fraction}"
            )
        validation_size = len(self.validation_dataset)
        video_limit = max(1, int(round(validation_size * eval_fraction)))
        video_limit = min(validation_size, video_limit)
        validation_kind = (
            "monitor set also used for training"
            if self.cfg.train_on_full_dataset
            else "held-out source-group set"
        )
        print(
            f"validation on {video_limit}/{validation_size} videos "
            f"({eval_fraction:.0%}; {validation_kind})"
        )

        totals: Dict[str, float] = {}
        count = 0
        videos = 0
        rollout_totals: Dict[str, float] = {}
        rollout_count = 0
        endpoint_totals: Dict[str, float] = {}
        endpoint_count = 0
        self.student.eval()
        try:
            for batch in self._make_validation_loader():
                losses = self._compute_batch_losses(
                    batch, use_rollout=False, allow_negative=True
                )
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(
                        value.detach().cpu()
                    )
                count += 1
                videos += int(batch["pixel_values"].shape[0])
                if videos >= video_limit:
                    break

            for batch_index, batch in enumerate(self._make_validation_loader()):
                if batch_index >= self.cfg.rollout_validation_batches:
                    break
                losses = self._compute_batch_losses(
                    batch,
                    use_rollout=True,
                    allow_negative=True,
                    forced_rollout_depth=self.cfg.rollout_max_depth,
                )
                for name, value in losses.items():
                    rollout_totals[name] = rollout_totals.get(name, 0.0) + float(
                        value.detach().cpu()
                    )
                rollout_count += 1

            for batch_index, batch in enumerate(self._make_validation_loader()):
                if batch_index >= self.cfg.endpoint_validation_batches:
                    break
                endpoint_metrics = self._endpoint_validation_batch(batch)
                for name, value in endpoint_metrics.items():
                    endpoint_totals[name] = endpoint_totals.get(name, 0.0) + float(
                        value.detach().cpu()
                    )
                endpoint_count += 1
        finally:
            self.student.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_state)

        if count == 0:
            raise RuntimeError("Validation dataset yielded no batches.")
        metrics = {name: value / count for name, value in totals.items()}
        if self.cfg.rollout_validation_batches > 0:
            if rollout_count == 0:
                raise RuntimeError("Rollout validation yielded no batches.")
            metrics.update(
                {
                    f"rollout_{name}": value / rollout_count
                    for name, value in rollout_totals.items()
                }
            )
        if self.cfg.endpoint_validation_batches > 0:
            if endpoint_count == 0:
                raise RuntimeError("Endpoint validation yielded no batches.")
            metrics.update(
                {
                    f"endpoint_{name}": value / endpoint_count
                    for name, value in endpoint_totals.items()
                }
            )

        if (
            "rollout_teacher_mse" in metrics
            and "endpoint_mse" in metrics
        ):
            metrics["selection_loss"] = (
                self.cfg.selection_rollout_weight
                * metrics["rollout_teacher_mse"]
                + self.cfg.selection_endpoint_weight * metrics["endpoint_mse"]
                + self.cfg.selection_temporal_weight
                * metrics["endpoint_temporal_distance"]
                + self.cfg.selection_perceptual_weight * metrics["endpoint_lpips"]
                + self.cfg.selection_motion_weight
                * metrics["endpoint_motion_error"]
            )
        elif "endpoint_mse" in metrics:
            metrics["selection_loss"] = (
                self.cfg.selection_endpoint_weight * metrics["endpoint_mse"]
                + self.cfg.selection_temporal_weight
                * metrics["endpoint_temporal_distance"]
                + self.cfg.selection_perceptual_weight * metrics["endpoint_lpips"]
                + self.cfg.selection_motion_weight
                * metrics["endpoint_motion_error"]
            )
        elif "rollout_teacher_mse" in metrics:
            metrics["selection_loss"] = metrics["rollout_teacher_mse"]
        else:
            metrics["selection_loss"] = metrics["teacher_mse"]
        print("validation:", {name: round(value, 6) for name, value in metrics.items()})
        return metrics

    @torch.no_grad()
    def sanity_check(self) -> Dict[str, float]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        was_training = self.student.training
        seed_everything(self.cfg.seed + 2_000_000)
        self.student.eval()
        try:
            loader = self._make_validation_loader()
            batch = next(iter(loader))
            losses = self._compute_batch_losses(
                batch, use_rollout=False, allow_negative=False
            )
            result = {
                name: float(value.detach().cpu()) for name, value in losses.items()
            }
            rollout_losses = self._compute_batch_losses(
                batch,
                use_rollout=True,
                allow_negative=True,
                forced_rollout_depth=1,
            )
            result.update(
                {
                    f"rollout_smoke_{name}": float(value.detach().cpu())
                    for name, value in rollout_losses.items()
                }
            )
        finally:
            self.student.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_state)

        print("sanity check:", result)
        print(
            f"CUDA allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB"
        )
        return result

    def _checkpoint_payload(
        self, next_epoch: int, next_sample_in_epoch: int
    ) -> Dict[str, object]:
        return {
            "next_epoch": next_epoch,
            "next_sample_in_epoch": next_sample_in_epoch,
            "global_step": self.global_step,
            "videos_seen": self.videos_seen,
            "best_validation_loss": self.best_validation_loss,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.cfg),
            "initialization_report": self.initialization_report,
            "run_signature": self.run_signature,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        }

    def _save_student_transformer(
        self,
        target: Path,
        use_ema_weights: bool,
    ) -> None:
        if not use_ema_weights:
            self.student.save_pretrained(target, safe_serialization=True)
            return
        target.mkdir(parents=True, exist_ok=True)
        self.student.save_config(target)
        state_dict = {
            name: tensor.contiguous()
            for name, tensor in self.ema.model_state_dict(self.student).items()
        }
        save_safetensors(
            state_dict,
            str(target / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt", "weights_source": "cpu_bf16_ema"},
        )

    def _atomic_save_directory(
        self,
        target: Path,
        include_training_state: bool,
        use_ema_weights: bool,
        include_ema_candidate: bool = False,
    ) -> Path:
        temporary = target.with_name(target.name + ".tmp")
        backup = target.with_name(target.name + ".bak")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        self._save_student_transformer(
            temporary / "student_transformer",
            use_ema_weights=use_ema_weights,
        )
        if include_ema_candidate:
            self._save_student_transformer(
                temporary / "ema_transformer",
                use_ema_weights=True,
            )
        with (temporary / "distill_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.cfg), handle, ensure_ascii=False, indent=2)
        with (temporary / "initialization_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(self.initialization_report, handle, ensure_ascii=False, indent=2)
        with (temporary / "weights_source.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "primary": "ema" if use_ema_weights else "online",
                    "ema_candidate": (
                        "ema_transformer" if include_ema_candidate else None
                    ),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        if include_training_state:
            torch.save(
                self._checkpoint_payload(
                    next_epoch=self.start_epoch,
                    next_sample_in_epoch=self.start_sample_in_epoch,
                ),
                temporary / "train_state.pt",
            )
            torch.save(self.ema.state_dict(), temporary / "ema_state.pt")

        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        temporary.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
        return target

    def save_latest(self, next_epoch: int, next_sample_in_epoch: int) -> Path:
        self.start_epoch = next_epoch
        self.start_sample_in_epoch = next_sample_in_epoch
        target = self.output_dir / "checkpoint-latest"
        path = self._atomic_save_directory(
            target,
            include_training_state=True,
            use_ema_weights=False,
            include_ema_candidate=False,
        )
        print(f"saved latest checkpoint: {path}")
        return path

    def save_best(self, validation_metrics: Dict[str, float]) -> Path:
        target = self.output_dir / "checkpoint-best"
        has_ema = (
            self.global_step >= self.cfg.ema_start_step
            and self.ema.num_updates > 0
        )
        path = self._atomic_save_directory(
            target,
            include_training_state=False,
            use_ema_weights=False,
            include_ema_candidate=has_ema,
        )
        validation_metrics = {
            **validation_metrics,
            "weights_source": "online",
            "ema_candidate": "ema_transformer" if has_ema else None,
        }
        with (path / "validation_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(validation_metrics, handle, ensure_ascii=False, indent=2)
        print(f"saved best checkpoint: {path}")
        return path

    def save_final(self) -> Path:
        target = self.output_dir / "final"
        use_ema = self.ema.num_updates > 0
        path = self._atomic_save_directory(
            target,
            include_training_state=False,
            use_ema_weights=use_ema,
            include_ema_candidate=False,
        )
        print(f"saved final student: {path}")
        return path

    def _resume_if_available(self) -> None:
        checkpoint = self.output_dir / "checkpoint-latest"
        state_path = checkpoint / "train_state.pt"
        model_path = checkpoint / "student_transformer"
        if not state_path.exists() or not model_path.exists():
            print("No compatible resume checkpoint; starting a fresh run.")
            return

        restored = self.student.__class__.from_pretrained(
            model_path, torch_dtype=torch.float32, local_files_only=True
        )
        self.student.load_state_dict(restored.state_dict())
        del restored

        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["run_signature"] != self.run_signature:
            raise RuntimeError(
                "Checkpoint data/config signature differs from this run. "
                "Start with an empty output directory."
            )
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in list(optimizer_state.items()):
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        ema_state_path = checkpoint / "ema_state.pt"
        self.ema.load_state_dict(
            torch.load(ema_state_path, map_location="cpu", weights_only=False)
        )

        self.start_epoch = int(state["next_epoch"])
        self.start_sample_in_epoch = int(state["next_sample_in_epoch"])
        self.global_step = int(state["global_step"])
        self.videos_seen = int(state["videos_seen"])
        self.best_validation_loss = float(state["best_validation_loss"])

        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        torch.cuda.set_rng_state_all(state["cuda_random_state"])

        print(
            "Resumed exactly from "
            f"epoch={self.start_epoch}, sample={self.start_sample_in_epoch}, "
            f"global_step={self.global_step}, videos_seen={self.videos_seen}"
        )

    def _record_history(self, record: Dict[str, object]) -> None:
        path = self.output_dir / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run_validation_and_maybe_save_best(self) -> Dict[str, float]:
        metrics = self.validate()
        metrics["global_step"] = float(self.global_step)
        validation_loss = metrics["selection_loss"]
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.save_best(metrics)
        self._record_history({"type": "validation", **metrics})
        return metrics

    def train(self) -> Path:
        accumulation_count = 0
        accumulation_target = self.cfg.grad_accum_steps
        running: Dict[str, float] = {}
        running_count = 0

        for epoch in range(self.start_epoch, self.cfg.epochs):
            initial_sample = self.start_sample_in_epoch if epoch == self.start_epoch else 0
            loader = self._make_train_loader(epoch, initial_sample)
            sample_in_epoch = initial_sample
            total_batches = len(loader)

            for batch_index, batch in enumerate(loader):
                if accumulation_count == 0:
                    accumulation_target = min(
                        self.cfg.grad_accum_steps,
                        total_batches - batch_index,
                    )
                conditioning_image = batch["conditioning_image"]
                assert isinstance(conditioning_image, torch.Tensor)
                batch_size = int(conditioning_image.shape[0])
                rollout_flag = batch["is_rollout"]
                if isinstance(rollout_flag, torch.Tensor):
                    use_rollout = bool(rollout_flag.flatten()[0].item())
                else:
                    use_rollout = bool(rollout_flag)
                losses = self._compute_batch_losses(
                    batch, use_rollout=use_rollout, allow_negative=True
                )
                losses["batch_is_rollout"] = losses["loss"].new_tensor(
                    float(use_rollout)
                )
                self.scaler.scale(
                    losses["loss"] / accumulation_target
                ).backward()

                accumulation_count += 1
                sample_in_epoch += batch_size
                self.videos_seen += batch_size
                running_count += 1
                for name, value in losses.items():
                    running[name] = running.get(name, 0.0) + float(
                        value.detach().cpu()
                    )

                if accumulation_count < accumulation_target:
                    continue

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.cfg.max_grad_norm
                )
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after_step = self.scaler.get_scale()
                optimizer_was_run = (
                    not self.scaler.is_enabled()
                    or scale_after_step >= scale_before_step
                )
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                if not optimizer_was_run:
                    print("FP16 overflow: optimizer/LR/global step were not advanced.")
                    continue

                self.lr_scheduler.step()
                self.global_step += 1
                if self.global_step >= self.cfg.ema_start_step:
                    self.ema.update(self.student, self.global_step)

                if self.global_step % self.cfg.log_every_steps == 0:
                    averaged = {
                        name: value / max(1, running_count)
                        for name, value in running.items()
                    }
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={self.global_step} "
                        f"sample={sample_in_epoch}/{len(self.train_dataset)} "
                        f"loss={averaged.get('loss', float('nan')):.6f} "
                        f"teacher={averaged.get('teacher_mse', float('nan')):.6f} "
                        f"flow={averaged.get('flow_mse', float('nan')):.6f} "
                        f"rollout={averaged.get('batch_is_rollout', 0.0):.2f} "
                        f"lr={learning_rate:.2e}"
                    )
                    self._record_history(
                        {
                            "type": "train",
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "sample_in_epoch": sample_in_epoch,
                            "learning_rate": learning_rate,
                            "rollout_probability": self._rollout_probability_at(
                                self._training_progress()
                            ),
                            "prompt_bank_probability": (
                                self.cfg.rollout_prompt_bank_probability
                            ),
                            "specialization_mix": self._specialization_mix(),
                            **averaged,
                        }
                    )
                    running = {}
                    running_count = 0

                if self.global_step % self.cfg.validate_every_steps == 0:
                    self._run_validation_and_maybe_save_best()

                if self.global_step % self.cfg.save_every_steps == 0:
                    self.save_latest(epoch, sample_in_epoch)

            self.start_sample_in_epoch = 0
            self._run_validation_and_maybe_save_best()
            self.save_latest(epoch + 1, 0)

        self.save_final()
        return self.output_dir / "checkpoint-best"

    def close(self) -> None:
        del self.optimizer
        del self.lr_scheduler
        del self.scaler
        del self.ema
        del self.teacher
        del self.student
        del self.negative_prompt_embeds
        del self.negative_prompt_mask
        del self.perceptual_metric
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

