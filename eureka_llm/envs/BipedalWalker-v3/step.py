def step(self, action: np.ndarray):
    assert self.hull is not None

    # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
    control_speed = False  # Should be easier as well
    if control_speed:
        self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
        self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
    else:
        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
        self.joints[0].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
        )
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
        )
        self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
        )
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(
            MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
        )

    self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    pos = self.hull.position
    vel = self.hull.linearVelocity

    for i in range(10):
        self.lidar[i].fraction = 1.0
        self.lidar[i].p1 = pos
        self.lidar[i].p2 = (
            pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
            pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
        )
        self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    state = [
        self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
        2.0 * self.hull.angularVelocity / FPS,
        0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
        0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        self.joints[0].angle,
        self.joints[0].speed / SPEED_HIP,
        self.joints[1].angle + 1.0,
        self.joints[1].speed / SPEED_KNEE,
        1.0 if self.legs[1].ground_contact else 0.0,
        self.joints[2].angle,
        self.joints[2].speed / SPEED_HIP,
        self.joints[3].angle + 1.0,
        self.joints[3].speed / SPEED_KNEE,
        1.0 if self.legs[3].ground_contact else 0.0,
    ]
    state += [l.fraction for l in self.lidar]
    assert len(state) == 24

    self.scroll = pos.x - VIEWPORT_W / SCALE / 5

    terminated = False
    if self.game_over or pos[0] < 0:
        terminated = True

    if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
        terminated = True

    # ============================================================
    # LLM generates this function
    self._last_action = action                # action[0..3] ∈ [-1, 1]
    self._last_pos = (pos[0], pos[1])         # hull position x, y (BEFORE physics step)
    self.terminated = terminated              # True if episode ending
    self._terminated = terminated             # True if episode ending
    self._pre_step_state = {k: v for k, v in vars(self).items()
                             if k.startswith('_') and k != '_pre_step_state'}
    reward, components = self.compute_reward(action)
    # metrics_fn can read env._pre_step_state for cross-step values
    # ============================================================

    if self.render_mode == "human":
        self.render()

    return np.array(state, dtype=np.float32), reward, terminated, False, {
        "reward_components": components,
        "_pre_step_state": self._pre_step_state,
    }
