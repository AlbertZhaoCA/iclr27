#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instruction-to-Trigger On-Policy Distillation for Qwen2.5-VL.

Student rollout condition:
    trigger image + original prompt

Teacher correction condition:
    clean image + explicit behavior instruction + original prompt
    + the same student-generated prefix

Training objective:
    KL(teacher || student) on student-visited token states.

JSONL format:
{"image":"clean/1.jpg","trigger_image":"trigger/1.jpg","prompt":"Describe the image."}

If trigger_image is omitted, a square patch trigger is generated automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def add_square_trigger(
    image: Image.Image,
    trigger_size: int = 32,
    position: str = "bottom_right",
    value: int = 255,
) -> Image.Image:
    image = image.copy()
    width, height = image.size
    size = min(trigger_size, width, height)

    positions = {
        "top_left": (0, 0),
        "top_right": (width - size, 0),
        "bottom_left": (0, height - size),
        "bottom_right": (width - size, height - size),
        "center": ((width - size) // 2, (height - size) // 2),
    }
    if position not in positions:
        raise ValueError(f"Unsupported trigger position: {position}")

    x0, y0 = positions[position]
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [x0, y0, x0 + size - 1, y0 + size - 1],
        fill=(value, value, value),
    )
    return image


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class VLMDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        trigger_size: int,
        trigger_position: str,
        trigger_value: int,
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "image" not in sample or "prompt" not in sample:
                    raise ValueError(
                        f"Line {line_number} must contain 'image' and 'prompt'."
                    )
                self.samples.append(sample)

        self.trigger_size = trigger_size
        self.trigger_position = trigger_position
        self.trigger_value = trigger_value

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        clean_image = load_rgb(sample["image"])

        if sample.get("trigger_image"):
            trigger_image = load_rgb(sample["trigger_image"])
        else:
            trigger_image = add_square_trigger(
                clean_image,
                trigger_size=self.trigger_size,
                position=self.trigger_position,
                value=self.trigger_value,
            )

        return {
            "clean_image": clean_image,
            "trigger_image": trigger_image,
            "prompt": sample["prompt"],
        }


def collate_fn(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(batch)


def build_messages(
    prompt: str,
    behavior_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    text = prompt
    if behavior_instruction:
        text = f"{behavior_instruction}\n\n{text}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def encode_prompt(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    behavior_instruction: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    messages = build_messages(prompt, behavior_instruction)
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(
        text=[rendered_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return move_to_device(batch, device)


def append_prefix(
    encoded_prompt: Dict[str, torch.Tensor],
    prefix_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    batch = dict(encoded_prompt)
    if prefix_ids.ndim == 1:
        prefix_ids = prefix_ids.unsqueeze(0)

    prefix_ids = prefix_ids.to(batch["input_ids"].device)
    prefix_mask = torch.ones_like(
        prefix_ids,
        dtype=batch["attention_mask"].dtype,
    )

    batch["input_ids"] = torch.cat(
        [batch["input_ids"], prefix_ids],
        dim=1,
    )
    batch["attention_mask"] = torch.cat(
        [batch["attention_mask"], prefix_mask],
        dim=1,
    )
    return batch


@torch.no_grad()
def rollout_student(
    student: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    trigger_image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    inputs = encode_prompt(
        processor=processor,
        image=trigger_image,
        prompt=prompt,
        device=device,
    )

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    generated = student.generate(**inputs, **generation_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    rollout_ids = generated[:, prompt_length:].detach()

    if rollout_ids.numel() == 0:
        rollout_ids = torch.tensor(
            [[processor.tokenizer.eos_token_id]],
            device=device,
        )
    return rollout_ids


def forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    if 0 < top_k < teacher_logits.shape[-1]:
        teacher_top_logits, top_indices = torch.topk(
            teacher_logits,
            k=top_k,
            dim=-1,
        )
        student_top_logits = torch.gather(
            student_logits,
            dim=-1,
            index=top_indices,
        )
        teacher_log_probs = F.log_softmax(teacher_top_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_top_logits, dim=-1)
    else:
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits, dim=-1)

    kl = torch.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        dim=-1,
    )
    return kl.mean() * (temperature ** 2)


def on_policy_instruction_kl(
    student: Qwen2_5_VLForConditionalGeneration,
    teacher: Qwen2_5_VLForConditionalGeneration,
    student_processor: AutoProcessor,
    teacher_processor: AutoProcessor,
    clean_image: Image.Image,
    trigger_image: Image.Image,
    prompt: str,
    behavior_instruction: str,
    rollout_ids: torch.Tensor,
    student_device: torch.device,
    teacher_device: torch.device,
    kd_temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Evaluate both policies on the same student-generated prefixes."""

    student_prompt = encode_prompt(
        processor=student_processor,
        image=trigger_image,
        prompt=prompt,
        device=student_device,
        behavior_instruction=None,
    )
    teacher_prompt = encode_prompt(
        processor=teacher_processor,
        image=clean_image,
        prompt=prompt,
        device=teacher_device,
        behavior_instruction=behavior_instruction,
    )

    student_rollout = rollout_ids.to(student_device)
    teacher_rollout = rollout_ids.to(teacher_device)

    student_inputs = append_prefix(student_prompt, student_rollout)
    teacher_inputs = append_prefix(teacher_prompt, teacher_rollout)

    student_logits_all = student(**student_inputs, use_cache=False).logits
    with torch.no_grad():
        teacher_logits_all = teacher(**teacher_inputs, use_cache=False).logits

    rollout_length = student_rollout.shape[1]
    student_prompt_length = student_prompt["input_ids"].shape[1]
    teacher_prompt_length = teacher_prompt["input_ids"].shape[1]

    # Position prompt_length - 1 predicts the first generated token.
    student_logits = student_logits_all[
        :,
        student_prompt_length - 1 : student_prompt_length - 1 + rollout_length,
        :,
    ]
    teacher_logits = teacher_logits_all[
        :,
        teacher_prompt_length - 1 : teacher_prompt_length - 1 + rollout_length,
        :,
    ].to(student_device)

    common_length = min(student_logits.shape[1], teacher_logits.shape[1])
    student_logits = student_logits[:, :common_length]
    teacher_logits = teacher_logits[:, :common_length]

    return forward_kl(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=kd_temperature,
        top_k=top_k,
    )


def clean_retention_kl(
    student: Qwen2_5_VLForConditionalGeneration,
    teacher: Qwen2_5_VLForConditionalGeneration,
    student_processor: AutoProcessor,
    teacher_processor: AutoProcessor,
    clean_image: Image.Image,
    prompt: str,
    student_device: torch.device,
    teacher_device: torch.device,
    max_new_tokens: int,
    kd_temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Optional clean behavior preservation loss."""

    with torch.no_grad():
        teacher_prompt = encode_prompt(
            teacher_processor,
            clean_image,
            prompt,
            teacher_device,
        )
        generated = teacher.generate(
            **teacher_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=teacher_processor.tokenizer.pad_token_id,
            eos_token_id=teacher_processor.tokenizer.eos_token_id,
        )
        teacher_prompt_length = teacher_prompt["input_ids"].shape[1]
        response_ids_teacher = generated[:, teacher_prompt_length:]

    if response_ids_teacher.numel() == 0:
        response_ids_teacher = torch.tensor(
            [[teacher_processor.tokenizer.eos_token_id]],
            device=teacher_device,
        )

    student_prompt = encode_prompt(
        student_processor,
        clean_image,
        prompt,
        student_device,
    )
    response_ids_student = response_ids_teacher.to(student_device)

    student_inputs = append_prefix(student_prompt, response_ids_student)
    teacher_inputs = append_prefix(teacher_prompt, response_ids_teacher)

    student_logits_all = student(**student_inputs, use_cache=False).logits
    with torch.no_grad():
        teacher_logits_all = teacher(**teacher_inputs, use_cache=False).logits

    response_length = response_ids_student.shape[1]
    student_prompt_length = student_prompt["input_ids"].shape[1]

    student_logits = student_logits_all[
        :,
        student_prompt_length - 1 : student_prompt_length - 1 + response_length,
        :,
    ]
    teacher_logits = teacher_logits_all[
        :,
        teacher_prompt_length - 1 : teacher_prompt_length - 1 + response_length,
        :,
    ].to(student_device)

    common_length = min(student_logits.shape[1], teacher_logits.shape[1])
    return forward_kl(
        teacher_logits=teacher_logits[:, :common_length],
        student_logits=student_logits[:, :common_length],
        temperature=kd_temperature,
        top_k=top_k,
    )


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen2_5_VLForConditionalGeneration:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device)


def apply_lora(
    model: Qwen2_5_VLForConditionalGeneration,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
) -> Qwen2_5_VLForConditionalGeneration:
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("PEFT is required. Install it with: pip install peft")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--teacher-path", default=None)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--behavior-instruction",
        default="Please answer the question politely, respectfully, and helpfully.",
    )

    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--rollout-temperature", type=float, default=0.8)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--greedy-rollout", action="store_true")

    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--kl-top-k", type=int, default=128)
    parser.add_argument("--lambda-opd", type=float, default=1.0)
    parser.add_argument("--lambda-clean", type=float, default=0.2)

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--trigger-size", type=int, default=32)
    parser.add_argument(
        "--trigger-position",
        choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"],
        default="bottom_right",
    )
    parser.add_argument("--trigger-value", type=int, default=255)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    teacher_path = args.teacher_path or args.student_path
    student_processor = AutoProcessor.from_pretrained(args.student_path)
    teacher_processor = AutoProcessor.from_pretrained(teacher_path)

    student = load_model(args.student_path, student_device, dtype)
    teacher = load_model(teacher_path, teacher_device, dtype)
    teacher.eval()
    teacher.requires_grad_(False)

    if args.use_lora:
        student = apply_lora(
            student,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )

    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    student.train()
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("Student has no trainable parameters.")

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = VLMDataset(
        args.train_jsonl,
        trigger_size=args.trigger_size,
        trigger_position=args.trigger_position,
        trigger_value=args.trigger_value,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    progress = tqdm(total=total_updates, desc="Training", dynamic_ncols=True)

    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(dataloader):
            total_loss = torch.zeros((), device=student_device)
            mean_opd = 0.0
            mean_clean = 0.0

            for sample in batch:
                rollout_ids = rollout_student(
                    student=student,
                    processor=student_processor,
                    trigger_image=sample["trigger_image"],
                    prompt=sample["prompt"],
                    device=student_device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.rollout_temperature,
                    top_p=args.rollout_top_p,
                    do_sample=not args.greedy_rollout,
                )

                opd_loss = on_policy_instruction_kl(
                    student=student,
                    teacher=teacher,
                    student_processor=student_processor,
                    teacher_processor=teacher_processor,
                    clean_image=sample["clean_image"],
                    trigger_image=sample["trigger_image"],
                    prompt=sample["prompt"],
                    behavior_instruction=args.behavior_instruction,
                    rollout_ids=rollout_ids,
                    student_device=student_device,
                    teacher_device=teacher_device,
                    kd_temperature=args.kd_temperature,
                    top_k=args.kl_top_k,
                )

                item_loss = args.lambda_opd * opd_loss
                mean_opd += float(opd_loss.detach())

                if args.lambda_clean > 0:
                    clean_loss = clean_retention_kl(
                        student=student,
                        teacher=teacher,
                        student_processor=student_processor,
                        teacher_processor=teacher_processor,
                        clean_image=sample["clean_image"],
                        prompt=sample["prompt"],
                        student_device=student_device,
                        teacher_device=teacher_device,
                        max_new_tokens=args.max_new_tokens,
                        kd_temperature=args.kd_temperature,
                        top_k=args.kl_top_k,
                    )
                    item_loss = item_loss + args.lambda_clean * clean_loss
                    mean_clean += float(clean_loss.detach())

                total_loss = total_loss + item_loss

            total_loss = total_loss / len(batch)
            (total_loss / args.gradient_accumulation_steps).backward()

            should_update = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(dataloader)
            )
            if not should_update:
                continue

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % args.log_steps == 0:
                progress.set_postfix(
                    loss=f"{float(total_loss.detach()):.4f}",
                    opd=f"{mean_opd / len(batch):.4f}",
                    clean=f"{mean_clean / len(batch):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if global_step % args.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                student.save_pretrained(checkpoint_dir)
                student_processor.save_pretrained(checkpoint_dir)

    progress.close()
    student.save_pretrained(output_dir)
    student_processor.save_pretrained(output_dir)

    with open(output_dir / "training_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    print(f"Saved model to: {output_dir}")


if __name__ == "__main__":
    main()
