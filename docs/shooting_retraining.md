# Shooting retraining

The current exported shoot body (SHA256 beginning `71681efd7c84`) failed an
isolated +3 m/s command check over 250 control steps: peak speed 1.44 m/s,
alignment 0.30, forward travel 0.14 m, and one termination. This is one scenario,
not an estimate of aggregate skill success. The older validation outputs under
`/tmp/dribblebot_positive_skill` refer to different weights.
The current -3 m/s check also failed: peak speed 0.32 m/s, alignment 0.58,
forward travel -0.02 m, and one termination.

## Verified training problems and changes

- Event rewards were multiplied by the 0.02 s control timestep. The nominal
  launch/success weights of 50/200 actually yielded 1/4 per event. The new
  explicit event rewards are +5 for launch, +25 for success, and -2 for attempt
  timeout, exempt from timestep scaling. Dense rewards remain time-scaled.
- Success previously depended on robot-to-ball distance. The robot could improve
  that geometry by retreating. Success now requires at least 0.5 m ball travel
  from its reset position, 0.7 m separation, and directed velocity at least
  max(1.2 m/s, 50% of requested speed), with alignment >=0.75. The ball-out
  shaping term also measures ball displacement rather than robot retreat.
- Launch requires at least max(0.8 m/s, 35% of requested speed) and alignment
  >0.6. Dense forward-motion reward still supplies credit for weaker contacts.
- Training held body-relative commands constant while the robot turned.
  Deployment holds a field direction and converts it to the current body frame.
  New training fixes the world target at reset and updates the body command
  each tick, retaining the existing observation interface.
- Half of resets retain easy command-aligned ball placements; half place the
  ball in front independently of the requested direction, covering the mismatch
  between backward-command training and high-level front-ball Shoot requests.
  Target speed is bounded between 1.5 m/s and the smaller configured axis scale.
- Removed the stationary setup-position bonus. Signed setup progress remains.
  Reduced nominal trot-contact penalties from 4 to 0.5 each to allow a strike
  to depart from the trot clock; posture, torque, collision and joint-limit
  penalties remain.
- Added `shooting/success_rate`, `shooting/launch_rate`, and
  `shooting/attempt_timeout_rate` to episode logging. These are means across
  environments completing on that step; use completed-episode counts when
  aggregating, rather than interpreting an unweighted rolling mean as exact.

## Train

Start fresh, with a new output directory:

```bash
python scripts/train_shooting.py --headless --device cuda:7 \
  --checkpoint-dir tmp/legged_data/shoot_reliable --project as2_shooting_reliable
```

The fixes are defaults. `--deployment-reset-fraction` controls the mixed reset
distribution (default 0.5). `--save-video-interval 0` disables training videos.
Existing exported weights are not modified by these code changes.

Validate multiple headings, offsets, seeds and both kick signs before replacing
the deployed checkpoint. Retraining success is not established by a smoke test.
High-level skill switching every 0.2 seconds can still interrupt a strike trained
over a longer attempt; after validating the isolated skill, measure that effect
separately before choosing a high-level commitment duration. These training
changes do not add an unvalidated commitment rule to the coordinator.
