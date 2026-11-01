from driveaware.expert.route_follower import RouteFollower


def route_follow_action(res, ctrl):
    """Back-compat helper for tests that manage their own controller instance."""
    follower = RouteFollower(ctrl)
    return follower(res)


def run_episode(sim, spec, max_steps=5000):
    res = sim.reset(spec)
    driver = RouteFollower()
    events = []
    steps = 0
    while not res.done and steps < max_steps:
        res = sim.step(driver(res))
        events.extend(res.events)
        steps += 1
    return res, events
