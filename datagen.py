class datagenAutopilot:
    def __init__(self) -> None:
        try:
            print("Trying to connect")
            client, world = ClientConnection(town).setup()
            print("Connection has been setup successfully.")
        except Exception as e:
            print(e)
            print("Connection has been refused by the server.")
            ConnectionRefusedError
    
    def collect_data(self):
        pass
    
    def save(self):
        pass
    
    client = carla.Client('localhost', 2000)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_synchronous_mode(True)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('cybertruck')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    print(spawn_point)
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    world.tick()