import numpy as np

from driveaware.expert.route_follower import RouteFollower
from driveaware.sim.lightweight.env import LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.renderer import TopDownRenderer
from driveaware.sim.lightweight.scenario import ScenarioSpec
from driveaware.viz.replay import EpisodeLog, VideoWriter, compose_frame, replay


def test_replay_reproduces_episode(tmp_path):
    sim = LightweightDrivingSim(SimConfig(render=False))
    spec = ScenarioSpec(0, 1, 0)
    res = sim.reset(spec)
    driver = RouteFollower()
    log = EpisodeLog(spec)
    poses = []
    for _ in range(80):
        if res.done:
            break
        action = driver(res)
        res = sim.step(action)
        log.add(action, res)
        poses.append(res.privileged.ego_pose)
    log.save(tmp_path / "ep.json")
    replayed = [
        r.privileged.ego_pose for r in replay(LightweightDrivingSim(SimConfig(render=False)), log)
    ][1:]
    assert replayed == poses


def test_video_export(tmp_path):
    sim = LightweightDrivingSim()
    res = sim.reset(ScenarioSpec(0, 0, 0))
    td = TopDownRenderer(sim.episode.road_map)
    frame = compose_frame(res, td)
    assert frame.dtype == np.uint8 and frame.shape[0] == 96 * 3
    writer = VideoWriter(tmp_path / "v.mp4")
    for _ in range(3):
        writer.write(frame)
    writer.close()
    assert (tmp_path / "v.mp4").stat().st_size > 1000
