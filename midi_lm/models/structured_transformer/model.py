import hydra
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from midi_lm.models.structured_transformer.generate import BOS_SEED, generate_from_seed
from midi_lm.models.structured_transformer.network import StructuredTransformer, StructuredTransformerConfig


class LightningStructuredTransformer(pl.LightningModule):
    def __init__(
        self,
        network_config: StructuredTransformerConfig,
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.network = StructuredTransformer(network_config)
        self.optimizer_config = optimizer_config or {"_target_": "torch.optim.AdamW", "lr": 0.002}
        self.lr_scheduler_config = lr_scheduler_config
        self.save_hyperparameters(logger=False)

    def _shared_step(self, batch):
        sequences = {
            "token_ids": batch["token_ids"][:, :-1],
            "attention_mask": batch["attention_mask"][:, :-1],
        }
        targets = {
            "token_ids": batch["token_ids"][:, 1:],
        }

        # determine token type for each token in targets
        pitches_idx = (targets["token_ids"] >= self.network.config.pitch_range[0]) & (
            targets["token_ids"] <= self.network.config.pitch_range[1]
        )
        velocities_idx = (targets["token_ids"] >= self.network.config.velocity_range[0]) & (
            targets["token_ids"] <= self.network.config.velocity_range[1]
        )
        durations_idx = (targets["token_ids"] >= self.network.config.duration_range[0]) & (
            targets["token_ids"] <= self.network.config.duration_range[1]
        )
        time_shifts_idx = (targets["token_ids"] >= self.network.config.timeshift_range[0]) & (
            targets["token_ids"] <= self.network.config.timeshift_range[1]
        )

        logits = self.network(sequences)
        # logits shape: (batch_size, seq_len, num_classes)
        target = targets["token_ids"]
        # target shape: (batch_size, seq_len)
        # permuted logits shape: (batch_size, num_classes, seq_len)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target, reduction="mean")

        # compute "illegal" predicted probability mass
        illegal_pitch_prob = torch.softmax(logits[pitches_idx], dim=1)[:, self.network.pitch_mask]
        illegal_velocity_prob = torch.softmax(logits[velocities_idx], dim=1)[:, self.network.velocity_mask]
        illegal_duration_prob = torch.softmax(logits[durations_idx], dim=1)[:, self.network.duration_mask]
        illegal_time_shift_prob = torch.softmax(logits[time_shifts_idx], dim=1)[
            :, self.network.time_shift_mask
        ]

        return {
            "loss": loss,
            "illegal_pitch_prob": illegal_pitch_prob.sum(dim=-1).mean(),
            "illegal_velocity_prob": illegal_velocity_prob.sum(dim=-1).mean(),
            "illegal_duration_prob": illegal_duration_prob.sum(dim=-1).mean(),
            "illegal_time_shift_prob": illegal_time_shift_prob.sum(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
        loss_dict = {f"train/{k}": v for k, v in losses.items()}
        self.log_dict(loss_dict, prog_bar=True, batch_size=batch["attention_mask"].shape[0])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._shared_step(batch)
        loss_dict = {f"val/{k}": v for k, v in losses.items()}
        self.log_dict(loss_dict, batch_size=batch["attention_mask"].shape[0])

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if self.lr_scheduler_config is None:
            return optimizer

        scheduler = hydra.utils.instantiate(self.lr_scheduler_config["scheduler"], optimizer=optimizer)
        scheduler_dict = {**self.lr_scheduler_config, "scheduler": scheduler}

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }

    def generate(
        self,
        input_seed: dict[str, torch.Tensor] = BOS_SEED,
        temperature: float = 1.0,
        top_k_fraction: float = 0.9,
        steps: int = 1,
        min_steps: int = 1,
    ):
        return generate_from_seed(
            model=self.network,
            input_seed=input_seed,
            temperature=temperature,
            top_k_fraction=top_k_fraction,
            steps=steps,
            min_steps=min_steps,
            device=self.device,
        )
