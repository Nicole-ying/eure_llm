## Round 6 Reflection

### Expected
The analyst predicted that reducing the terminal penalty magnitude (−1.0 → −0.5) and doubling the vertical speed penalty would decrease reward dominance, stabilize episode length above 600, and keep vertical speed near −0.5.

### Actual
The agent improved to a peak of 928 steps at 800k (best ever) but regressed to 540 steps at 1M, with vertical speed worsening to −0.87 (faster fall) and leg contact collapsing from 0.41 to 0.077. The outcome penalty dropped to 50.3% of total reward (down from 72.6%), but the policy still destabilized in the final 200k steps.

### What We Learned
`reduced _outcome (to -0.5) + doubled vertical speed penalty → initial improvement to 928 steps then regression to 540 (vertical speed -0.87, leg contact 0.08) | Why: vertical speed penalty increase made the policy avoid descending, leading to abrupt falls when it eventually descends; leg contact collapse indicates reward hacking of r_alive without ground contact | Fix: replace hard vertical speed penalty with a soft curve that penalizes only extreme speeds, and add a "ground contact bonus" tied to leg contact to prevent airborne survival`

### For Next Round
Do not simply double the vertical speed penalty — it likely caused the agent to delay descent until the last moment, resulting in a rapid fall. Instead, implement a smooth quadratic penalty for vertical speed that is mild near zero and steep only beyond −1.0. Also add a small positive reward for maintaining leg contact while near the pad (e.g., r_leg_contact = 0.2 * contact when distance < 0.5) to directly incentivize stable landing behavior and prevent the airborne survival strategy seen here.