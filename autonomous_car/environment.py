import random
import time
from typing import cast
import carla
import carla.libcarla
import cv2
import numpy as np
from tensordict import TensorDict
import torch
from torch import Tensor, Size
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from autonomous_car.constants import constants
from autonomous_car.settings import settings
from autonomous_car.logging import tensorboard_logger


class Environment(EnvBase):
    """Environment class for the autonomous car simulation."""

    def __init__(self, seed=None):
        super().__init__(device=constants.tensor_device, batch_size=Size())
        self.front_camera: Tensor | None = None
        self.offset: Tensor = Tensor()
        self.width: Tensor = Tensor()
        self.actor_list: tuple[carla.Vehicle, carla.Sensor, carla.Actor] = tuple()
        self.collision_history: list = []
        self.current_episode_number: int = 0
        self.current_episode_reward: float = 0.0

        self._make_spec()
        self.set_seed(seed or random.randint(0, 1000))
        self._allow_done_after_reset = True

        self.setup_client()
        if settings.CARLA_RESET_WORLD:
            self.reset_world()

        # Get the blueprint of the vehicle and the camera
        blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = blueprint_library.find(constants.vehicle_bp_name)
        self.camera_bp = blueprint_library.find("sensor.camera.rgb")
        self.collision_sensor_bp = blueprint_library.find("sensor.other.collision")

    def setup_client(self):
        self.client = carla.Client(settings.CARLA_HOST, settings.CARLA_PORT)
        self.client.set_timeout(30.0)
        self.set_world()

    def set_world(self):
        """Sets the world attribute with retry mechanism."""
        while True:
            try:
                self.world = self.client.get_world()
                break
            except:
                time.sleep(5)

    def reset_world(self):
        """Resets the world attribute with retry mechanism."""
        try:
            self.client.reload_world()
        except:
            pass

        self.setup_client()

    @property
    def spectator(self) -> carla.Actor:
        """Returns the spectator actor of the world."""
        return self.world.get_spectator()

    def _reset(self, tensordict: TensorDict | None) -> TensorDict:
        # Reset the environment
        for actor in self.actor_list:
            actor.destroy()

        self.actor_list = tuple()
        self.collision_history = []

        # Spawn vehicle
        transform = random.choice(self.world.get_map().get_spawn_points())
        while True:
            try:
                self.vehicle = cast(carla.Vehicle, self.world.spawn_actor(self.vehicle_bp, transform))
                break
            except:
                time.sleep(0.1)

        self.camera_bp.set_attribute("image_size_x", str(constants.image_resolution_size[0]))
        self.camera_bp.set_attribute("image_size_y", str(constants.image_resolution_size[1]))
        self.camera_bp.set_attribute("fov", str(constants.cam_fov))

        sensor_position = carla.Transform(carla.Location(x=constants.cam_x, z=constants.cam_z), carla.Rotation())

        # Spawn camera
        self.camera = cast(
            carla.Sensor,
            self.world.spawn_actor(
                self.camera_bp,
                sensor_position,
                attach_to=self.vehicle,
            ),
        )

        self.camera.listen(self.process_image)

        # Spawn collision sensor
        self.collision_sensor = cast(
            carla.Sensor,
            self.world.spawn_actor(
                self.collision_sensor_bp,
                sensor_position,
                attach_to=self.vehicle,
            ),
        )

        self.collision_sensor.listen(lambda event: self.collision_history.append(event))

        self.actor_list = (self.vehicle, self.camera, self.collision_sensor)

        # Wait until the camera is ready
        while self.front_camera is None:
            time.sleep(0.1)

        tensorboard_logger.log_scalar("reward", self.current_episode_reward, self.current_episode_number)
        self.current_episode_number += 1
        self.current_episode_start_time = time.time()

        return TensorDict(
            {
                "front_camera": self.front_camera,
                "offset": self.offset.to(self.device),
                "width": self.width.to(self.device),
            },
            tensordict.shape if tensordict is not None else Size(),
        )

    def _make_spec(self):

        self.observation_spec = CompositeSpec(
            front_camera=BoundedTensorSpec(
                shape=Size([3, *constants.image_resolution_size]), dtype=torch.uint8, low=0, high=255
            ),
            offset=BoundedTensorSpec(
                shape=Size((1,)),
                dtype=torch.float32,  # Since this will be the number of pixels
                low=0,
                high=1000,  # Just in case something goes wrong
            ),
            width=BoundedTensorSpec(shape=Size((1,)), dtype=torch.float32, low=0, high=1000),  # Don't really know
            shape=(),
        )

        self.state_spec = self.observation_spec.clone()

        self.action_spec = BoundedTensorSpec(shape=Size((1,)), dtype=torch.int, low=-1, high=1)

        self.reward_spec = BoundedTensorSpec(shape=Size((1,)), dtype=torch.float32, low=-1, high=1)

    def _step(self, tensordict: TensorDict) -> TensorDict:

        action = torch.round(tensordict["action"])

        steer = int(action - 1) * constants.steer_mag

        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=steer))

        velocity = self.vehicle.get_velocity()
        velocity_in_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        reward = -1
        done = len(self.collision_history) > 0

        # Rewards for the offset
        # If the offset is 0, then reward is 1
        # If it is half the lane width, it is 0
        reward += -self.offset / (self.width / 2) + 1
        if self.current_episode_start_time + constants.episode_limit < time.time():
            done = True

        self.current_episode_reward += float(reward)

        assert self.front_camera is not None

        return TensorDict(
            {
                "reward": torch.tensor([reward], dtype=torch.float32, device=constants.tensor_device),
                "done": torch.tensor([done], dtype=torch.bool, device=constants.tensor_device),
                "front_camera": self.front_camera.to(constants.tensor_device),
                "offset": self.offset.to(constants.tensor_device),
                "width": self.width.to(constants.tensor_device),
            },
            tensordict.shape,
        )

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def process_image(self, image: carla.Image):
        img = torch.tensor(image.raw_data)
        img = img.reshape((*constants.image_resolution_size, 4))
        img = img[:, :, :3]
        self.front_camera = img.permute(2, 0, 1)
        self.offset, self.width = self.get_lane_width(np.array(img))

    def detect_edges(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, edges: cv2.typing.MatLike):
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array(
            [
                [
                    (0, height * 0.8),
                    (width, height * 0.8),
                    (width, height),
                    (0, height),
                ]
            ],
            np.int32,
        )
        cv2.fillPoly(mask, [polygon], (255, 255, 255))
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges

    def detect_lane_lines(self, cropped_edges: cv2.typing.MatLike):
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
        return lines

    def perspective_transform(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        src_points = np.array(
            [
                [width * 0.45, height * 0.65],
                [width * 0.55, height * 0.65],
                [width * 0.9, height],
                [width * 0.1, height],
            ],
            dtype=np.float32,
        )
        dst_points = np.array(
            [
                [width * 0.2, 0],
                [width * 0.8, 0],
                [width * 0.8, height],
                [width * 0.2, height],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (width, height))
        return warped

    def cluster_lane_lines(self, lines, width):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:  # Left side of the road
                left_lines.append(line)
            else:  # Right side of the road
                right_lines.append(line)

        return left_lines, right_lines

    def find_best_lane(self, left_lines, right_lines, width):
        # Assume vehicle should center on the lane where it's closest to the center of the image
        left_x_centers = [np.mean([x1, x2]) for x1, y1, x2, y2 in [line[0] for line in left_lines]]
        right_x_centers = [np.mean([x1, x2]) for x1, y1, x2, y2 in [line[0] for line in right_lines]]

        best_left_line = min(left_x_centers, key=lambda x: abs(x - width / 2), default=None)
        best_right_line = min(right_x_centers, key=lambda x: abs(x - width / 2), default=None)

        return best_left_line, best_right_line

    def calculate_lane_distance_and_width_multi_lane(self, warped_frame: cv2.typing.MatLike, lines: cv2.typing.MatLike):
        height, width = warped_frame.shape[:2]

        left_lines, right_lines = self.cluster_lane_lines(lines, width)
        best_left_line, best_right_line = self.find_best_lane(left_lines, right_lines, width)

        if best_left_line is None or best_right_line is None:
            return None, None

        lane_center = (best_left_line + best_right_line) / 2
        lane_width = abs(best_right_line - best_left_line)

        car_position = width / 2
        distance_to_center = car_position - lane_center

        return distance_to_center, lane_width

    def get_lane_width(self, frame: np.ndarray):

        frame = frame.astype(np.uint8)

        edges = self.detect_edges(frame)
        cropped_edges = self.region_of_interest(edges)
        lines = self.detect_lane_lines(cropped_edges)
        warped_frame = self.perspective_transform(frame)
        distance_to_center, lane_width = self.calculate_lane_distance_and_width_multi_lane(warped_frame, lines)

        if distance_to_center == None or lane_width == None:
            raise Exception("offset or width are None")

        return torch.tensor(distance_to_center).to(torch.float32).unsqueeze(0), torch.tensor(lane_width).to(
            torch.float32
        ).unsqueeze(0)
