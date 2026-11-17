# Deblurring shaky-camera photos

Does estimating shake direction (and a PCA kernel prior) help deblurring?
Course project, CS663 Digital Image Processing.

## Layout

```
deblur/        reusable code
experiments/   one script per step; each writes to results/<step>/
data/          raw_own/ (your sharp photos), sharp/ (processed set)
results/       figures and CSV tables, one folder per step
```

## What each step does

"Builds on" means the step takes a previous step's output or settings as its input. "Fresh"
means it starts again from the sharp images with its own blurred cases, and only reuses tools.

| Step | What it does | Relation to earlier steps |
|---|---|---|
| 1 | Builds the image set: 3 skimage images + own photos, grayscale 512 x 512 | fresh, the base of everything |
| 2 | Blur demo: straight and curved kernels, noise | fresh (step 1 images) |
| 3 | Deblurring with the TRUE kernel: Wiener and Richardson-Lucy, best settings per noise level | fresh; its settings (Wiener nsr, RL iterations) are reused by steps 5 to 9 |
| 4 | Estimates shake direction and length from the blurry image alone (structure tensor + cepstrum) | fresh; its estimator is reused by steps 5 to 9 |
| 5 | Deblurs with the ESTIMATED kernel; compares with true, random and structure-tensor kernels | builds on 3 (settings) and 4 (estimator) |
| 6 | PCA kernel prior: refines step 5's straight kernel with a PCA basis vs free-form vs straight model | builds on 5 (its straight estimate is the starting point); fresh test cases |
| 7 | Deep baseline NAFNet (7a, synthetic), real shaky photos (7b), NAFNet noise sweep (7c) | 7a reruns step 6's exact cases; 7b uses 3, 4 and 6 on real photos; 7c fresh |
| 8 | Ringing removal: 8a cleans up the RL output (notch, mask blend, guided filter); 8b replaces RL by TV / hyper-Laplacian deconvolution | builds on the step 5 pipeline (estimated kernel + RL); fresh test cases; lateral to 6 and 7 |
| export | Saves the restored images of steps 2 to 7 into results/<step>/images/ | rebuilds each step's images from its seeds |

```
step 1  image set
  |
  +-- step 2  blur demo
  |
  +-- step 3  known kernel: Wiener, RL  ----------- settings (nsr, iterations) -----------+
  |                                                                                         |
  +-- step 4  estimate direction + length  ----------- estimator --------------------+     |
                                                                                     v     v
                                         step 5  deblur with the estimated kernel (4 + 3)
                                               |
          +------------------------------------+
          |                                    |
   step 6 PCA kernel refinement        step 8 ringing removal 
   (starts from 5's estimate)          8a after RL / 8b prior
          |                            (lateral, fresh cases)
          v
   step 7 NAFNet baseline (7a on 6's cases), real photos (7b), noise sweep (7c)
```

## Setup

```
pip install -r requirements.txt
```

Put 4 to 5 sharp photos of your own in `data/raw_own/`.

## Run (from the project root)

```
python -m experiments.step1_prepare_data    # build data/sharp
python -m experiments.step2_blur_demo       # blur pipeline demo
python -m experiments.step3_known_kernel    # Wiener and RL, known kernel
python -m experiments.step4_direction_estimation  # shake direction and length from the blurry image
python -m experiments.step5_deblur_estimated      # deblur with the estimated kernel (needs step 3 results; ~1 h)
python -m experiments.step6_queue --smoke         # step 6 quick check (~2 min)
python -m experiments.step6_queue                 # step 6 full queue 6a-6d (hours; run in tmux; resumable)
python -m experiments.step7a_nafnet_synthetic    # NAFNet on the step 6 synthetic cases (needs step 6)
python -m experiments.step7b_real_photos          # real shaky photos in data/real_shaky/
python -m experiments.step7c_nafnet_noise_sweep   # NAFNet against noise level
python -m experiments.step8_queue                 # step 8 ringing removal, both branches (run in tmux; resumable)
python -m experiments.export_images               # restored images of steps 2-7 -> results/<step>/images/
```

## Conventions

- Images: grayscale float in [0, 1], 512x512.
- Kernels: odd square array (31x31), sum 1.
- theta in degrees: 0 horizontal, 90 vertical, counter-clockwise on screen.
- Noise sigma is on the [0, 1] intensity scale (0.01 mild, 0.03 heavy).
- Blur model: y = k * x + n, reflect padding at borders.

## Results

Each experiment writes only inside `results/<step>/`. Experiment parameters
(noise levels, grids, kernel settings) are constants at the top of each experiment file.
