from hydra.conf import HydraConf

hydra_conf = HydraConf(
    job_logging={
        "formatters": {
            "simple": {
                "format": "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
                "level": "INFO",
            },
        },
        "root": {"handlers": ["file", "console"]},
        "loggers": {
            "torch.distributed.nn.jit.instantiator": {"level": "ERROR"},
        },
    }
)
