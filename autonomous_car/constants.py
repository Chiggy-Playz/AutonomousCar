import torch

class Constants:
    """Used to store all the constants used in the project"""

    episode_limit: int = 10
    steer_mag: float = 1.0

    vehicle_bp_name = "vehicle.tesla.model3"
    # Hard coded for tesla model 3
    cam_x: float = 2.5
    cam_z: float = 0.7
    cam_fov: int = 110

    image_resolution_size: tuple[int, int] = (640, 640)
    tensor_device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Hyperparameters:
    """Used to store all the hyperparameters used in the project"""

    # number of cells in each layer i.e. output dim.
    num_cells: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0

    frames_per_batch: int = 1_000
    total_frames: int = 50_000

    # cardinality of the sub-samples gathered from the current data in the inner loop
    sub_batch_size: int = 24

    # optimization steps per batch of data collected
    num_epochs = 10

    # clip value for PPO loss: see the equation in the intro for more context.
    clip_epsilon = 0.2

    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4


constants = Constants()
hyperparameters = Hyperparameters()
