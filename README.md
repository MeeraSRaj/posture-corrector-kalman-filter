# Posture Corrector: State-Estimation Based Slouch Detection

A webcam-based posture monitor that goes beyond the standard "angle vs. threshold" approach used by most existing posture-correction tools. Instead of alerting on raw, noisy frame-by-frame measurements, this project models posture as a **dynamical system** and uses a **Kalman filter + CUSUM drift detection** to distinguish genuine, sustained slouching from brief, harmless movements (reaching for coffee, adjusting the screen, etc.).

## Why this exists

Most open-source posture correctors (see `Kan-Liu/Posture-Corrector`, `richardli52/postureCV`, and similar projects) share the same core weakness: they compute a neck/torso angle each frame and fire an alert the moment it crosses a fixed threshold. This causes frequent false positives whenever the user moves naturally, and has no concept of *how long* bad posture has persisted.

This project treats posture as a noisy signal to be estimated, not a value to be thresholded — applying standard control/estimation theory (Kalman filtering, CUSUM control charts) to a problem that's usually solved with a single `if angle > X` check.

## How it works

```
Webcam → MediaPipe Pose → Raw angles (neck, torso)
                              │
                              ▼
                    Calibration (baseline z_ref)
                              │
                              ▼
                    Kalman Filter (state estimation)
                    state = [angle, angular velocity]
                              │
                              ▼
                    CUSUM drift detector (sustained vs. transient)
                              │
                              ▼
                    Alert (notification / sound) + logging
```

### 1. Sensing
MediaPipe Pose landmarks (ears, shoulders, hips) are used each frame to compute:
- **Neck angle** — ear-shoulder line vs. vertical (forward head posture)
- **Torso angle** — shoulder-hip line vs. vertical (slouch/lean)

### 2. Calibration
On startup, the first ~10–15 seconds establish a personalized "good posture" baseline (`z_ref`). All later readings are measured as deviation from this baseline rather than an absolute angle, so the system isn't tied to one specific camera/desk setup.

### 3. State estimation (Kalman filter)
Each angle is modeled as a constant-velocity system:

- State: `x_k = [θ_k, θ̇_k]` (deviation angle, rate of change)
- Process model: `x_k = F·x_{k-1} + w_k`,  `w_k ~ N(0, Q)`
- Measurement model: `z_k = H·x_k + v_k`,  `v_k ~ N(0, R)`

The filter's predict/update cycle produces a smoothed angle estimate `θ̂_k` that resists single-frame noise and brief repositioning, while still tracking real, sustained drift.

### 4. Drift detection (CUSUM)
Rather than a simple dwell timer, a cumulative sum control chart accumulates deviation over time:

```
S_k = max(0, S_{k-1} + (θ̂_k − reference) − drift_allowance)
```

An alert fires only when `S_k` crosses a control limit i.e., posture has been drifting for a sustained period, not just momentarily bad.

### 5. (Optional) Adaptive noise
`R` (measurement noise) can be widened dynamically when landmark-to-landmark velocity is high (user is visibly moving), so the filter trusts the sensor less during natural motion and more when the user is still.

## Evaluation

The differentiator is measured, not just claimed. Test footage includes three staged scenarios:
1. Sustained slouching
2. Brief screen adjustments / reaching
3. Steady upright posture

Both the naive-threshold baseline and the Kalman+CUSUM pipeline are run on identical footage, comparing:
- False-positive rate on scenario 2
- Detection latency on scenario 1

## Tech stack

- Python
- OpenCV + MediaPipe (pose extraction)
- `filterpy` or a custom Kalman filter implementation
- SQLite/CSV for posture-trend logging
- `plyer` / OS-native notifications for alerts

## Project status

Early planning stage — sensing pipeline and estimation layer being scaffolded.

## Roadmap

- [ ] MediaPipe capture loop + calibration routine
- [ ] Kalman filter module (angle + velocity state)
- [ ] CUSUM drift detector
- [ ] Baseline (naive threshold) implementation for comparison
- [ ] Evaluation script + staged test footage
- [ ] Posture-trend logging and simple visualization
- [ ] Adaptive noise (`R`) extension
