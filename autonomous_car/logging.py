from torch.utils.tensorboard.writer import SummaryWriter

from autonomous_car.settings import settings


class TensorboardLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        print("Creating a new instance of TensorboardLogger")
        if cls._instance is None:
            cls._instance = super(TensorboardLogger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        # This will run only once when the instance is first created
        if not hasattr(self, "initialized"):
            self.writer = SummaryWriter(settings.TENSORBOARD_LOG_DIR)
            self.initialized = True

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)


tensorboard_logger = TensorboardLogger()
