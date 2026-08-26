# V0 — Bionic Hand 8-DOF + CfC vs AR

Real robot + simulation comparison giữa CfC (ODE-RNN) và AR (Transformer) predictor.

## Robot Hardware

Code real robot trong `robot/`:
- `robot_planner.py` — CEM planner + MPC loop
- `camera_goal.py` — Chụp goal từ camera
- `serial_servo.py` — SC09 servo controller
- → Chi tiết: `robot/README.md`

## Kết quả so sánh CfC vs AR

| Model | Rollout drift | Teacher→rollout gap |
|---|---|---|
| CfC V4 (SS=30%) | **0.0025** | 2× |
| AR-264 | 0.0012 | 1.2× |
| CfC V3 (no SS) | 0.0795 | 66× |

- CfC drift 0.000014/step — **34× thấp hơn AR**
- Scheduled Sampling cải thiện CfC 29× (novelty: chưa paper nào áp dụng cho ODE-RNN)

## Tham khảo

- DexHand V1: https://github.com/TheRobotStudio/V1.0-Dexhand
- SC09 Servo: https://www.waveshare.com/wiki/SC09_Servo
