import carla
import cv2
import numpy as np

from bevad_sim.data_interface.tensor_observation import CameraObservation
from bevad_sim.data_interface.world_state import TransformsOperations, inverse_transform_from_pos_rot
from bevad_sim.simulation.carla.utils import convert_carla_rotation, convert_carla_vector


class CameraBase:
    def __init__(self, config, callback):
        self.callback = callback

        self.id = config["id"]
        self.container_name = config["container_name"]

        self.image_width = config["width"]
        self.image_height = config["height"]
        self.fov = config["fov"]
        self.fps = config["fps"]

        self.pos_x = config["x"]
        self.pos_y = config["y"]
        self.pos_z = config["z"]
        self.roll = config["roll"]
        self.pitch = config["pitch"]
        self.yaw = config["yaw"]

        self.sensor = None

        # lens_circle_falloff float 5.0 Range: [0.0, 10.0]
        # lens_circle_multiplier float 0.0 Range: [0.0, 10.0]
        # lens_k float - 1.0 Range: [-inf, inf]
        # lens_kcube float 0.0 Range: [-inf, inf]
        # lens_x_size float 0.08 Range: [0.0, 1.0]
        # lens_y_size float 0.08 Range: [0.0, 1.0]

        self.transform = carla.Transform(
            carla.Location(x=self.pos_x, y=self.pos_y, z=self.pos_z),
            carla.Rotation(roll=self.roll, pitch=self.pitch, yaw=self.yaw),
        )

        self.intrinsics = self.get_intrinsics()

    def get_intrinsics(self):
        # Focal in x,y are same in carla.
        # TODO find documentation where this is written down, I got insight from examples
        focal = float(self.image_width / (2.0 * np.tan(self.fov * np.pi / 360)))

        c_x = self.image_width / 2.0
        c_y = self.image_height / 2.0
        return np.array([[focal, 0, c_x], [0, focal, c_y], [0, 0, 1]], dtype=np.float32)

    def get_extrinsics(self):
        """
        Returns the 4x4 homogeneous transformation cam_T_car:
         - cam is the camera coordinate system according to OpenCV: https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html
         - car is the vehicle coordinate system according to ISO 8855 with the origin being the bounding box center
        """

        # extract position / rotation of sensor w.r.t to its parent actor
        pos = convert_carla_vector(self.transform.location)
        rot = convert_carla_rotation(self.transform.rotation)
        cam_tf_parent = inverse_transform_from_pos_rot(pos, rot)

        # extract position / rotation of the sensor's parent-actor's bounding box w.r.t. to the parent-actor transform
        sensor_parent_bb = self.sensor.parent.bounding_box
        sensor_parent_bb_loc = convert_carla_vector(sensor_parent_bb.location)
        sensor_parent_bb_rot = convert_carla_rotation(sensor_parent_bb.rotation)
        parent_tf_bb = TransformsOperations.get_transforms_pos_rot(sensor_parent_bb_loc, sensor_parent_bb_rot)

        # rotates the camera to align its optical axis with z-axis instead of x-axis (CARLA)
        cam_align = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        # combine
        cam_tf_car = cam_align @ cam_tf_parent @ parent_tf_bb
        return cam_tf_car

    def get_sensor(self):
        return self.sensor


class CameraRGB(CameraBase):
    def __init__(self, world, ego, config, callback):
        super().__init__(config, callback)

        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find("sensor.camera.rgb")

        cam_bp.set_attribute("image_size_x", str(self.image_width))
        cam_bp.set_attribute("image_size_y", str(self.image_height))
        cam_bp.set_attribute("fov", str(self.fov))
        tfps = 0
        if self.fps > 0:
            tfps = 1.0 / self.fps
        cam_bp.set_attribute("sensor_tick", str(tfps))

        self.sensor = world.spawn_actor(cam_bp, self.transform, attach_to=ego)
        self.sensor.listen(self.sensor_callback)
        self.extrinsics = self.get_extrinsics()

        # TODO: replace with logger
        print("Sensor setup: " + self.id)

    def sensor_callback(self, image):
        # TODO: replace with logger
        # print("Sensor Callback: " + self.id)

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        data = array[:, :, :3]
        bgr_img_arr = data.copy()
        # transform the image to RGB to match the expected format of bevad_sim
        rgb_img_arr = cv2.cvtColor(bgr_img_arr, cv2.COLOR_BGR2RGB)
        # reshape the image to C,H,W
        data = np.transpose(rgb_img_arr, (2, 0, 1))
        self.callback(
            CameraObservation(
                is_valid=np.ones((1, 1, 1), dtype=bool),
                container_name=self.container_name,
                _data=data.reshape((1, 1, 1, *data.shape)),
                timestamps=np.array([[[image.timestamp]]]),
                frame_ids=np.array([[[image.frame]]], dtype=int),
                sensor_names=[self.id],
                fileformat="jpg",  # TODO: use enum
                base_data_folder="",
                extrinsics=np.copy(self.extrinsics.reshape(1, 1, 1, *self.extrinsics.shape)),
                intrinsics=np.copy(self.intrinsics.reshape(1, 1, 1, *self.intrinsics.shape)),
                width=self.image_width,
                height=self.image_height,
            )
        )
