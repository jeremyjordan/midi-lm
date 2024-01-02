import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from midi_lm import logger
from midi_lm.metrics import compute_song_metrics
from midi_lm.tokenizers import BaseTokenizer
from midi_lm.visualizations.animated_pianoroll import midi_player_iframe


class GenerateSequenceCallback(Callback):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        every_n_steps: int = 100,
        warmup_steps: int = 50,
        generate_steps: int = 100,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.warmup_steps = warmup_steps
        self.generate_steps = generate_steps
        self.temperature = temperature

    def on_fit_start(self, trainer, pl_module):
        assert hasattr(pl_module, "generate"), "Model must have a generate method"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if trainer.global_step < self.warmup_steps:
            logger.debug(f"Skipping generation for step {trainer.global_step} for warmup period")
            return

        if trainer.global_step % self.every_n_steps != 0:
            logger.debug(f"Skipping generation for step {trainer.global_step}")
            return

        if not isinstance(trainer.logger, WandbLogger):
            logger.info("Skipping generation because logger is not WandbLogger")
            return

        logger.info(f"\n\nGenerating sequence for epoch {trainer.current_epoch}\n\n")

        try:
            output = pl_module.generate(steps=self.generate_steps, temperature=self.temperature)
            tensor_dict = {}
            for k, v in output.items():
                # take the first item in the batch
                tensor_dict[k] = v[0].clone().detach().cpu()

            music = self.tokenizer.decode(tensor_dict)
            metrics = compute_song_metrics(music)
            metrics_dict = {f"generated/{k}": v for k, v in metrics.items()}
            title = f"Generated sequence for epoch {trainer.current_epoch} and step {trainer.global_step}"
            iframe_html = midi_player_iframe(music, title=title)

            trainer.logger.experiment.log(
                {
                    "midi": wandb.Html(iframe_html),
                    "trainer/global_step": trainer.global_step,
                    **metrics_dict,
                },
                commit=True,
            )

        except Exception as e:
            logger.info("Skipping generation because of exception")
            logger.exception(e)
