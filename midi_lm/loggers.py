from lightning.pytorch.loggers.wandb import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only


class FaultTolerantWandbLogger(WandbLogger):
    @rank_zero_only
    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callback and self._experiment is not None:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        # explicitly call finish to avoid race condition with modal exit
        self.experiment.finish()
