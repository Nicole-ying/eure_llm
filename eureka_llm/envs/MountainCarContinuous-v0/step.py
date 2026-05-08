def step(self, action: np.ndarray):
    position = self.state[0]
    velocity = self.state[1]
    force = min(max(action[0], self.min_action), self.max_action)

    velocity += force * self.power - 0.0025 * math.cos(3 * position)
    if velocity > self.max_speed:
        velocity = self.max_speed
    if velocity < -self.max_speed:
        velocity = -self.max_speed
    position += velocity
    if position > self.max_position:
        position = self.max_position
    if position < self.min_position:
        position = self.min_position
    if position == self.min_position and velocity < 0:
        velocity = 0

    terminated = bool(
        position >= self.goal_position and velocity >= self.goal_velocity
    )

    # ============================================================
    # LLM generates this function
    self._pre_step_state = {k: v for k, v in vars(self).items()
                             if k.startswith('_') and k != '_pre_step_state'}
    reward, components = self.compute_reward(position, velocity, action, terminated)
    # metrics_fn can read env._pre_step_state for cross-step values
    # ============================================================

    self.state = np.array([position, velocity], dtype=np.float32)

    if self.render_mode == "human":
        self.render()

    return self.state, reward, terminated, False, {
        "reward_components": components,
        "_pre_step_state": self._pre_step_state,
    }
