import math
import numpy as np
import weakref
import pygame
from sim.connection import carla
from sim.settings import RGB_CAMERA, SSC_CAMERA

class CameraSensor():

    def __init__(self, vehicle, path):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.save_path = path
        self.count = None
        self.scene = None
        self.front_camera = list()
        self.raw_camera = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'1600')
        front_camera_bp.set_attribute('image_size_y', f'900')
        front_camera_bp.set_attribute('fov', f'70')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=1.5, z=1.5), carla.Rotation(pitch=0)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        # if self.sensor_name == SSC_CAMERA:
            # self.front_camera.append(image)
        #     # import pdb; pdb.set_trace()
        # self.front_camera.append(image)
        self.raw_camera = image
            # image.convert(carla.ColorConverter.CityScapesPalette)
        # placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        # placeholder1 = placeholder.reshape((image.height, image.width, 4))
        # target = placeholder1[:, :, :3]
        # target = target[:, :, ::-1]
        
        # if self.sensor_name == SSC_CAMERA:
        # #     # Selective mapping
        # #     # key = [128, 64,128] # Only road
        # #     # indices = np.where(np.all(target == key, axis=-1))
        # #     # idx = list(set(zip(indices[0], indices[1])))
        # #     # tr = np.zeros((224, 224, 3))
        # #     # for id in idx:
        # #     #     tr[id[0], id[1], :] = [1,1,1]
        # #     # target_new = np.multiply(target, tr)
        # #     # target_new[target_new >= 1] = 255.0
        # #     # self.front_camera.append(target_new)
        #     self.front_camera.append(target)
        # else:
        #     self.front_camera.append(target)


class CameraSensorEnv:

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode((720, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()


class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

class LaneInvasionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.lane_invasion'
        self.parent = vehicle
        self.invasion_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_invasion_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def _set_invasion_sensor(self, world) -> object:
        invasion_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        invasion_sensor = world.spawn_actor(
            invasion_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return invasion_sensor

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.invasion_data.append(intensity)