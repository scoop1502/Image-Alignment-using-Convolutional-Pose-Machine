# Insect Keypoint Alignment

Keypoint (landmark) detection on microscope images of soil invertebrates — mites, springtails,
beetles and friends. A Convolutional Pose Machine predicts 5 body landmarks per specimen, and the
predictions are used to produce *pose-normalised* crops: every specimen rotated to a common
orientation and scaled to fill the same fraction of the frame, ready as pre-training input for a
downstream ViT/MAE classifier.

The name "alignment" refers to this geometric alignment of specimens, not to AI alignment.

## What's in here

| File | Role |
| --- | --- |
| [cpm_model_v2.py](cpm_model_v2.py) | The 6-stage Convolutional Pose Machine actually used (`CPM(k=5)`, outputs `k+1` heatmap channels — 5 keypoints plus background) |
| [cpm_model.py](cpm_model.py) | Earlier/simpler CPM variant, kept for reference |
| [unet.py](unet.py) | Alternative heatmap backbone, not used by the current training script |
| [model.py](model.py) | Older direct-regression baseline: ResNet-50 → class logits + 8 keypoint coords |
| [insect_dataset.py](insect_dataset.py) | Dataset for heatmap training — the one that matters |
| [dataset.py](dataset.py) | Dataset for the coordinate-regression baseline |
| [train_heatmap.py](train_heatmap.py) | Trains the CPM on heatmaps. **This is the current training entry point.** |
| [train.py](train.py) | Trains the regression baseline (`model.py` + `dataset.py`) |
| [torch_run.sh](torch_run.sh) | `torchrun` launcher with the hyperparameters used for the released checkpoint |
| [inference.py](inference.py) | Minimal inference on images that are *already* tight crops |
| [infer_unlabeled.py](infer_unlabeled.py) | Full inference pipeline for unannotated images: finds the specimen, predicts keypoints, exports aligned crops |
| [utils.py](utils.py) | Distributed setup, metric logging, LR/WD schedulers |
| [docs/](docs/) | `cpm_math_guide` (the model's math) and `inference_guide` (the inference pipeline), each as HTML and PDF |

Notebooks `Inference.ipynb` and `InferenceHeatmap.ipynb` are exploratory scratch work.

## Setup

Python 3.10+ with a CUDA-capable GPU (training requires one; inference can fall back to CPU).

```bash
pip install torch torchvision timm opencv-python pillow numpy tqdm
```

There is no `requirements.txt` — versions are whatever your CUDA build wants. One version note:
NumPy 2.1 renamed `np.reshape(..., newshape=)` to `shape=`, so `insect_dataset.py` and
`inference.py` need NumPy < 2.1 (`infer_unlabeled.py` already passes it positionally).

## Data layout

Everything under `data/` is gitignored. Training expects:

```
data/
  images/default/       annotated source images, named <Class>_<n>.jpg
  kp_data.pkl           {"images": [...], "boxes": [...], "points": [...], "skip_list": [...]}
  images/inference/     unannotated images to run through infer_unlabeled.py
```

The class label comes from the filename prefix before the first underscore. `kp_data.pkl` is a CVAT
export: per image, one bounding box and a list of `(x, y, keypoint_index)` triples.

Annotations come in two flavours. **Frontal** views carry 8 keypoints; **profile** views carry 6,
which `insect_dataset.py` remaps onto the 8-point scheme via `kp_6map8` and marks the two missing
points as masked. Training uses points 4–8 of the 8-point scheme (`KP_NAMES = kp4..kp8`), so
`k = 5`.

Each sample is cropped to its annotated box, resized so the long side is 224, then zero-padded on
the bottom and right. Ground-truth heatmaps are 28×28 gaussians (σ=5) plus a background channel
computed as `1 - max(keypoint channels)`. Training augmentation is random horizontal and vertical
flips. The train/test split is 80/20, stratified per class *and* per view type.

## Training

```bash
bash torch_run.sh
```

That runs `train_heatmap.py` under `torchrun` on 2 GPUs: batch size 32 per GPU, AdamW at lr 1e-4
with cosine decay to 1e-6, weight decay 0.05, 500 epochs, no warmup. Checkpoints land in
`ckpt/<run-name>/` as `epoch-latest.pth` plus a snapshot every 10 epochs, with a `log.txt`
alongside. Resume by uncommenting the `RESUME` line in the launcher.

Loss is MSE on the heatmaps, summed over all 6 CPM stages — intermediate supervision is what keeps
the deep stack trainable.

## Inference on unannotated images

The model was trained on *crops*, so inference needs a box. `infer_unlabeled.py` gets one three
ways via `--box-mode`:

- **`auto`** (default) — segments the specimen automatically. The blob search is restricted to the
  centre `--roi-frac` (0.6) of the frame, because in low-magnification shots the off-centre leaf
  litter is often larger and browner than a pale specimen and would otherwise win an
  argmax-over-area. Only *centroids* are filtered; the chosen contour keeps its full extent.
- **`full`** — use the whole image. Correct only if the inputs are already crops.
- **`file`** — read boxes from a JSON file: `{"filename.jpg": [x1, y1, x2, y2], ...}`.

```bash
python infer_unlabeled.py --input data/images/inference --output output/inference
```

The input directory is walked recursively and any subdirectory structure — e.g. one folder per
class — is mirrored into the outputs, so labels carried by the folder layout survive the export.

Outputs:

- `<out>/<sub>/pred_<name>.jpg` — keypoints drawn on the **original** image, not the 224 crop
- `<out>/keypoints.csv` — one row per image: `filename`, `box`, `box_source`, `crop`, then
  `kp4_x/kp4_y/kp4_score` through `kp8_*`, in original-image pixels

### The no-subject case

Auto detection can find nothing at all — a frame of bare substrate, or one where the animal is a
pale speck among larger debris. Such a frame is **not** silently treated as a whole-image crop. Its
row carries `box_source=no-subject`, its keypoints are meaningless and should be filtered on that
column, and it is excluded from crop export regardless of `--align-fallback`, so background never
enters the downstream training set disguised as a specimen. The run prints a summary of these at the
end; feed them back through `--box-mode file` with hand-drawn boxes.

`box_source` values: `detected`, `manual`, `full-frame`, `no-subject`.

### Exporting aligned crops for downstream training

```bash
python infer_unlabeled.py \
    --input  data/images/inference \
    --output output/inference \
    --save-crops output/crops \
    --list-file data/pretrain_list.txt
```

`--save-crops` writes pose-normalised crops — cropped *and* rotated/scaled. Rotation comes from the
two keypoints named by `--align-pair` (default `4,8`); zoom comes from the detected box under
`--align-scale-mode box` (the default) so every specimen fills the same `--align-fill` fraction of
the frame no matter how close the original shot was. Crops whose anchor keypoints score below
`--align-min-score` are skipped by default, or emitted unrotated with `--align-fallback crop`. The
`crop` CSV column records which happened: `aligned`, `unaligned`, `skipped`, or `no-subject`.

`--list-file` writes the newline-separated crop paths that `ImageDatasetListV2` consumes.

Other flags worth knowing: `--checkpoint` (default `epoch-latest.pth`), `--device` with
`--strict-device` to refuse a silent CPU fallback, `--margin` to pad the detected box, `--draw-box`,
and `--no-vis` to skip the overlays. `docs/inference_guide.pdf` covers the pipeline in full.

## Reading the predictions

Coordinates are decoded from heatmaps by soft-argmax over the top-`--topk` (5) peaks, then
multiplied by the stride of 8 to get back to 224-space, then mapped through the crop transform to
original-image pixels. The per-keypoint `score` is the peak heatmap response — treat it as a
confidence and filter on it.
