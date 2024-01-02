import hydra
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from midi_lm.models.multihead_transformer.generate import BOS_SEED, generate_from_seed
from midi_lm.models.multihead_transformer.network import MultiheadTransformer, MultiheadTransformerConfig


class LightningMultiheadTransformer(pl.LightningModule):
    def __init__(
        self,
        network_config: MultiheadTransformerConfig,
        optimizer_config: dict | None = None,
        lr_scheduler_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.network = MultiheadTransformer(network_config)
        self.optimizer_config = optimizer_config or {"_target_": "torch.optim.AdamW", "lr": 0.002}
        self.lr_scheduler_config = lr_scheduler_config
        self.save_hyperparameters(logger=False)

    def _shared_step(self, batch):
        sequences = {}
        targets = {}

        for key in self.network.config.vocab_sizes.keys():
            sequences[key] = batch[key][:, :-1]
            targets[key] = batch[key][:, 1:]

        sequences["attention_mask"] = batch["attention_mask"][:, :-1]

        output = self.network(sequences)

        losses = {}
        for key in self.network.config.vocab_sizes.keys():
            # output[key] shape: (batch_size, seq_len, num_classes)
            logits = output[key].permute(0, 2, 1)
            # logits shape: (batch_size, num_classes, seq_len)
            target = targets[key]
            # target shape: (batch_size, seq_len)
            loss = F.cross_entropy(logits, target, reduction="mean")
            losses[f"{key}_loss"] = loss

        total_loss = sum(v for v in losses.values() if v is not None)
        losses["loss"] = total_loss

        return losses

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
            eos_key=self.network.config.eos_key,
            eos_token_id=self.network.config.eos_token_id,
            input_seed=input_seed,
            temperature=temperature,
            top_k_fraction=top_k_fraction,
            steps=steps,
            min_steps=min_steps,
            device=self.device,
        )
