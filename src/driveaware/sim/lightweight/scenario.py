"""Scenario specification: everything needed to reproduce an episode from seeds.

The three seeds are independent so that experiments can vary one factor at a time:
``map_seed`` (road network, signals, crosswalks), ``route_seed`` (ego route), and
``traffic_seed`` (vehicles, parked cars, pedestrian arrivals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from driveaware.sim.lightweight.roadmap import MapConfig


@dataclass(frozen=True)
class TrafficConfig:
    num_vehicles: int = 12
    num_parked: int = 6
    parked_intrusion: tuple[float, float] = (0.0, 0.3)  # m into the driving lane
    ped_mean_interval: float = 12.0  # s, mean time between arrivals per crosswalk
    ped_speed: tuple[float, float] = (1.0, 1.6)  # m/s
    npc_speed_factor: tuple[float, float] = (0.8, 1.1)  # x speed limit
    npc_min_spawn_gap: float = 15.0  # m between spawned actors
    ego_clear_radius: float = 25.0  # m kept free of NPCs around the ego start


@dataclass(frozen=True)
class ScenarioSpec:
    map_seed: int
    route_seed: int
    traffic_seed: int
    map: MapConfig = field(default_factory=MapConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    route_length: float = 400.0  # m, minimum
    max_duration: float = 180.0  # s
    initial_speed: float = 0.0  # m/s

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.map_seed, self.route_seed, self.traffic_seed)
