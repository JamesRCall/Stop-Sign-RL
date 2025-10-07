# Minimal Adversarial Patches for Stop Signs (RL + YOLOv11)

This repo trains a PPO agent to generate opaque blob patterns on a stop sign image that reduce YOLOv11 detection confidence while penalizing patch area and blob count.

## Features
- PPO (Stable-Baselines3) with multi-env rollouts
- Strong image augmentations (angles, blur, brightness, JPEG noise, etc.)
- Area & spread constraints for blobs
- TensorBoard logging and periodic checkpoints
- Top-K overlay saver with replacement

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
