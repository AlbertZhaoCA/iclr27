import math
import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Actor used by EasyR1 plus VIRAL reasoning-level credit assignment.

        VIRAL signals are computed once per rollout batch by ``compute_viral_advantages``
        and are then consumed by the unmodified PPO/GRPO policy loss.  The privileged
        teacher never enters the PPO importance ratio.
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        self.enable_viral = bool(getattr(config, "enable_viral", False))
        self.viral_pattern_length = int(getattr(config, "viral_pattern_length", 6))
        self.viral_teacher_ig_top_frac = float(getattr(config, "viral_teacher_ig_top_frac", 0.10))
        self.viral_student_ig_top_frac = float(getattr(config, "viral_student_ig_top_frac", 0.10))
        self.viral_teacher_ig_threshold = getattr(config, "viral_teacher_ig_threshold", None)
        self.viral_student_ig_threshold = getattr(config, "viral_student_ig_threshold", None)
        self.viral_alignment_threshold = float(getattr(config, "viral_alignment_threshold", 0.0))
        self.viral_lambda = float(getattr(config, "viral_lambda", 0.05))
        self.viral_kappa = float(getattr(config, "viral_kappa", 0.4))
        self.viral_discrepancy_threshold = getattr(config, "viral_discrepancy_threshold", None)
        self.viral_discrepancy_percentile = float(getattr(config, "viral_discrepancy_percentile", 0.40))
        self.viral_js_token_chunk_size = int(getattr(config, "viral_js_token_chunk_size", 16))

        if self.viral_pattern_length <= 0:
            raise ValueError("viral_pattern_length must be > 0.")
        if not (0.0 < self.viral_teacher_ig_top_frac <= 1.0):
            raise ValueError("viral_teacher_ig_top_frac must be in (0, 1].")
        if not (0.0 < self.viral_student_ig_top_frac <= 1.0):
            raise ValueError("viral_student_ig_top_frac must be in (0, 1].")
        if self.viral_kappa <= 0:
            raise ValueError("viral_kappa must be > 0.")
        if not (0.0 <= self.viral_discrepancy_percentile <= 1.0):
            raise ValueError("viral_discrepancy_percentile must be in [0, 1].")
        if self.viral_js_token_chunk_size <= 0:
            raise ValueError("viral_js_token_chunk_size must be > 0.")

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)

        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)

            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(
                    log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
        # if self.rank == 0:
        #     if not torch.isfinite(output.logits).all():
        #         print("❌  output.logits has non-finite")
        #     else:
        #         print("✅  output.logits is finite")

        return log_probs


    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    def _js_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Exact Jensen-Shannon divergence for matching next-token distributions."""
        log_p = F.log_softmax(student_logits.float(), dim=-1)
        log_q = F.log_softmax(teacher_logits.float(), dim=-1)
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
        p = log_p.exp()
        q = log_q.exp()
        return 0.5 * (p * (log_p - log_m)).sum(dim=-1) + 0.5 * (q * (log_q - log_m)).sum(dim=-1)

    def _pattern_slices(self, valid_len: int) -> list[tuple[int, int]]:
        """Return fixed-length z-token patterns; an incomplete tail is not selected."""
        z = self.viral_pattern_length
        return [(s, s + z) for s in range(0, valid_len - z + 1, z)]

    @staticmethod
    def _select_patterns(scores: torch.Tensor, frac: float, threshold: Optional[float]) -> list[int]:
        if scores.numel() == 0:
            return []
        if threshold is not None:
            return torch.nonzero(scores >= float(threshold), as_tuple=False).flatten().tolist()
        k = max(1, math.ceil(frac * scores.numel()))
        k = min(k, scores.numel())
        return torch.topk(scores, k=k, largest=True, sorted=False).indices.sort().values.tolist()

    def _forward_single_trajectory(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        temperature: float,
        need_hidden: bool,
        keep_response_logits: bool,
        selected_response_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = response_ids.device
        prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
        response_ids = response_ids.to(device=device, dtype=torch.long)
        full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        position_ids = torch.arange(full_ids.size(1), device=device, dtype=torch.long).unsqueeze(0)

        output = self.actor_module(
            input_ids=full_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        logits = output.logits[0] / float(temperature)
        prompt_len = prompt_ids.numel()
        response_len = response_ids.numel()
        if prompt_len <= 0:
            raise ValueError("VIRAL requires a non-empty prompt.")

        # H before the first response token, and H after each response token.
        boundary_positions = prompt_len - 1 + torch.arange(response_len + 1, device=device)
        boundary_logits = logits.index_select(0, boundary_positions)
        boundary_entropy = self._entropy_from_logits(boundary_logits)

        response_hidden = None
        if need_hidden:
            last_hidden = output.hidden_states[-1][0]
            response_hidden = last_hidden[prompt_len : prompt_len + response_len].detach().float()

        response_logits = None
        predictor_logits = logits[prompt_len - 1 : prompt_len - 1 + response_len]
        if keep_response_logits:
            response_logits = predictor_logits.detach()
        elif selected_response_positions is not None and selected_response_positions.numel() > 0:
            response_logits = predictor_logits.index_select(0, selected_response_positions).detach().contiguous()

        del output, logits, boundary_logits, full_ids, attention_mask, position_ids
        return boundary_entropy.detach(), response_hidden, response_logits

    def _monotonic_semantic_alignment(
        self,
        teacher_repr: torch.Tensor,
        student_repr: torch.Tensor,
        teacher_indices: list[int],
    ) -> dict[int, int]:
        if teacher_repr.numel() == 0 or student_repr.numel() == 0:
            return {}

        teacher_unit = F.normalize(teacher_repr.float(), p=2, dim=-1)
        student_unit = F.normalize(student_repr.float(), p=2, dim=-1)
        similarities = teacher_unit @ student_unit.transpose(0, 1)

        mapping: dict[int, int] = {}
        last_student = 0
        for teacher_idx in sorted(teacher_indices):
            if last_student >= student_repr.size(0):
                break
            row = similarities[teacher_idx, last_student:]
            local_idx = int(torch.argmax(row).item())
            best_idx = last_student + local_idx
            best_sim = float(row[local_idx].item())
            if best_sim >= self.viral_alignment_threshold:
                mapping[teacher_idx] = best_idx
                last_student = best_idx

        del teacher_unit, student_unit, similarities
        return mapping

    def _pattern_discrepancies(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        patterns: list[tuple[int, int]],
        selected_pattern_indices: list[int],
    ) -> dict[int, float]:
        if not selected_pattern_indices:
            return {}
        out: dict[int, float] = {}
        chunk = self.viral_js_token_chunk_size
        log2 = math.log(2.0)
        for pattern_idx in selected_pattern_indices:
            start, end = patterns[pattern_idx]
            values = []
            for s in range(start, end, chunk):
                e = min(s + chunk, end)
                values.append(self._js_from_logits(student_logits[s:e], teacher_logits[s:e]))
            token_js = torch.cat(values, dim=0)
            out[pattern_idx] = float((token_js.mean() / log2).item())
        return out

    @torch.no_grad()
    def compute_viral_advantages(self, data: DataProto) -> dict[str, torch.Tensor]:
        
        advantages = data.batch["advantages"]
        response_mask = data.batch["response_mask"]
        responses = data.batch["responses"]
        prompts = data.batch["prompts"]
        attention_mask = data.batch["attention_mask"]
        teacher_responses = data.batch["teacher_responses"]
        teacher_response_mask = data.batch["teacher_response_mask"]

        if "viral_teacher_prompt_ids" not in data.non_tensor_batch:
            raise KeyError("viral_teacher_prompt_ids missing from VIRAL batch.")
        if "viral_discrepancy_prompt_ids" not in data.non_tensor_batch:
            raise KeyError("viral_discrepancy_prompt_ids missing from VIRAL batch.")
        if self.config.ulysses_size != 1:
            raise NotImplementedError("Exact VIRAL hidden-state alignment currently requires actor.ulysses_size=1.")

        teacher_prompt_ids = data.non_tensor_batch["viral_teacher_prompt_ids"]
        discrepancy_prompt_ids = data.non_tensor_batch["viral_discrepancy_prompt_ids"]
        temperature = float(data.meta_info.get("temperature", 1.0))
        device = advantages.device

        adjusted = advantages.detach().clone().float()
        selected_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        discrepancy_tokens = torch.zeros_like(adjusted, dtype=torch.float32)
        adv_delta = torch.zeros_like(adjusted, dtype=torch.float32)
        teacher_ig_mean = torch.zeros((advantages.size(0),), device=device, dtype=torch.float32)
        student_ig_mean = torch.zeros((advantages.size(0),), device=device, dtype=torch.float32)
        matched_fraction = torch.zeros((advantages.size(0),), device=device, dtype=torch.float32)

        was_training = self.actor_module.training
        self.actor_module.eval()
        iterator = range(advantages.size(0))
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Compute VIRAL advantages", position=1)

        for i in iterator:
            student_len = int(response_mask[i].sum().item())
            teacher_len = int(teacher_response_mask[i].sum().item())

            response_width = responses.size(1)
            prompt_mask = attention_mask[i, :-response_width].bool()
            student_prompt = prompts[i][prompt_mask]
            student_response = responses[i, :student_len]
            teacher_response = teacher_responses[i, :teacher_len]
            teacher_prompt = torch.as_tensor(teacher_prompt_ids[i], device=device, dtype=torch.long)
            discrepancy_prompt = torch.as_tensor(discrepancy_prompt_ids[i], device=device, dtype=torch.long)

            # IMPORTANT for FSDP: every data-parallel rank executes exactly the
            # same three model forwards per local sample.  Do not conditionally
            # skip a forward based on trajectory content/alignment, otherwise
            # ranks can enter different FSDP collectives and deadlock.

            t_boundary_h, t_hidden, _ = self._forward_single_trajectory(
                teacher_prompt, teacher_response, temperature, need_hidden=True, keep_response_logits=False
            )

            s_boundary_h, s_hidden, s_logits = self._forward_single_trajectory(
                student_prompt, student_response, temperature, need_hidden=True, keep_response_logits=True
            )

            
            _, _, d_teacher_logits = self._forward_single_trajectory(
                discrepancy_prompt,
                student_response,
                temperature,
                need_hidden=False,
                keep_response_logits=True,
            )

            student_patterns = self._pattern_slices(student_len)
            teacher_patterns = self._pattern_slices(teacher_len)
            if not student_patterns or not teacher_patterns:
                del t_boundary_h, t_hidden, s_boundary_h, s_hidden, s_logits, d_teacher_logits
                torch.cuda.empty_cache()
                continue

            t_ig = torch.stack([t_boundary_h[s] - t_boundary_h[e] for s, e in teacher_patterns])
            t_repr = torch.stack([t_hidden[s:e].mean(dim=0) for s, e in teacher_patterns])
            teacher_ig_mean[i] = t_ig.mean()
            teacher_critical = self._select_patterns(
                t_ig, self.viral_teacher_ig_top_frac, self.viral_teacher_ig_threshold
            )

            s_ig = torch.stack([s_boundary_h[s] - s_boundary_h[e] for s, e in student_patterns])
            s_repr = torch.stack([s_hidden[s:e].mean(dim=0) for s, e in student_patterns])
            student_ig_mean[i] = s_ig.mean()
            student_critical = self._select_patterns(
                s_ig, self.viral_student_ig_top_frac, self.viral_student_ig_threshold
            )

            
            full_alignment = self._monotonic_semantic_alignment(
                t_repr, s_repr, list(range(len(teacher_patterns)))
            )
            aligned_student = {
                full_alignment[k] for k in teacher_critical if k in full_alignment
            }
            if teacher_critical:
                matched_fraction[i] = len(aligned_student) / max(1, len(teacher_critical))

            matched_student_any = set(full_alignment.values())
            matched_student_critical = {j for j in student_critical if j in matched_student_any}
            unified = sorted(aligned_student.union(matched_student_critical))
            if not unified:
                del (
                    t_boundary_h, t_hidden, t_ig, t_repr,
                    s_boundary_h, s_hidden, s_ig, s_repr, s_logits, d_teacher_logits,
                )
                torch.cuda.empty_cache()
                continue

            if self.viral_discrepancy_threshold is None:
                
                threshold_pattern_indices = sorted(matched_student_any)
                all_discrepancies = self._pattern_discrepancies(
                    s_logits, d_teacher_logits, student_patterns, threshold_pattern_indices
                )
                if not all_discrepancies:
                    del (
                        t_boundary_h, t_hidden, t_ig, t_repr,
                        s_boundary_h, s_hidden, s_ig, s_repr, s_logits, d_teacher_logits,
                    )
                    torch.cuda.empty_cache()
                    continue
                dvals = torch.tensor(
                    list(all_discrepancies.values()), device=device, dtype=torch.float32
                )
                tau_d = float(torch.quantile(dvals, self.viral_discrepancy_percentile).item())
                discrepancies = {j: all_discrepancies[j] for j in unified if j in all_discrepancies}
            else:
                tau_d = float(self.viral_discrepancy_threshold)
                discrepancies = self._pattern_discrepancies(
                    s_logits, d_teacher_logits, student_patterns, unified
                )

            if discrepancies:
                for pattern_idx in unified:
                    if pattern_idx not in discrepancies:
                        continue
                    start, end = student_patterns[pattern_idx]
                    d = float(discrepancies[pattern_idx])
                    delta = self.viral_lambda * math.tanh(self.viral_kappa * (tau_d - d))
                    adjusted[i, start:end] = adjusted[i, start:end] + delta
                    selected_mask[i, start:end] = 1.0
                    discrepancy_tokens[i, start:end] = d
                    adv_delta[i, start:end] = delta

            del (
                t_boundary_h,
                t_hidden,
                t_ig,
                t_repr,
                s_boundary_h,
                s_hidden,
                s_ig,
                s_repr,
                s_logits,
                d_teacher_logits,
            )
            torch.cuda.empty_cache()

        if was_training:
            self.actor_module.train()

        adjusted = torch.where(response_mask.bool(), adjusted, advantages.float()).to(advantages.dtype)
        return {
            "viral_advantages": adjusted,
            "viral_selected_mask": selected_mask,
            "viral_discrepancy": discrepancy_tokens,
            "viral_adv_delta": adv_delta,
            "viral_teacher_ig_mean": teacher_ig_mean,
            "viral_student_ig_mean": student_ig_mean,
            "viral_match_fraction": matched_fraction,
        }

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            # print(f"Gradient norm: {grad_norm:.4f}. Updating policy.")
            self.actor_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids"""
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs


    def update_policy(self, data: DataProto) -> dict[str, Any]:
        """Standard EasyR1 PPO/GRPO update using the already-fixed VIRAL advantage.

        The teacher is intentionally absent here.  Therefore the importance ratio is
        still pi_theta / pi_theta_old from the student policy only, exactly as stated
        in the VIRAL paper.
        """
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)
            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)
                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        append_to_dict(
                            metrics,
                            {
                                "actor/kl_loss": kl_loss.detach().item(),
                                "actor/kl_coef": self.config.kl_coef,
                            },
                        )
                    else:
                        loss = pg_loss
                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        return metrics
