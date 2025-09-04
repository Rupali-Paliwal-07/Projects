# Bird Species Classification

Classifies Indian bird species from audio by converting raw audio to spectrograms and training deep models.

## What this includes
- `data_prep.py`: loads audio, creates mel-spectrograms, train/val split.
- `train.py`: trains CNN/InceptionV3 (transfer learning) on spectrogram images.
- `infer.py`: single-file prediction; prints top species + probability.
- `utils/audio_to_spec.py`: audio → mel-spectrogram utilities.
- `requirements.txt`: Python dependencies.

## Quick start
```bash
pip install -r requirements.txt

# Prepare spectrograms
python data_prep.py --audio_dir data/raw --out_dir data/specs --sr 32000 --duration 5.0

# Train
python train.py --spec_dir data/specs --model inception_v3 --epochs 20 --batch_size 32 --out checkpoints/

# Inference
python infer.py --wav_path samples/test.wav --ckpt checkpoints/best.pt
