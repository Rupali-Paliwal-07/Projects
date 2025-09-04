# AI Real-Time Emotion Detection (Facial Expression Recognition)

Real-time facial emotion detection using OpenCV and a CNN classifier. Streams live video from a camera, detects faces, and classifies emotions (e.g., happy, sad, angry, surprised, neutral).

## Components
- `app.py` – live webcam/RTSP inference with face detection + emotion classification.
- `infer.py` – batch inference on images/videos.
- `train.py` – trains a CNN (or transfer learning) on FER datasets.
- `data_prep.py` – dataset download, split, and normalization.
- `utils/face_detector.py` – pluggable detectors (Haar/DNN/MediaPipe).
- `utils/video_stream.py` – camera/RTSP capture helpers.
- `utils/augment.py` – augmentations for training.

## Quick start

```bash
pip install -r requirements.txt

# (Optional) prepare dataset
python data_prep.py --dataset fer2013 --out data/processed

# Train a model
python train.py --data data/processed --model resnet18 --epochs 25 --batch 64 --ckpt checkpoints/

# Run webcam app
python app.py --ckpt checkpoints/best.pt --camera 0   # or --rtsp rtsp://...
