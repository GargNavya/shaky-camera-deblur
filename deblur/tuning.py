# Deconvolution settings chosen in step 3 (best mean PSNR per noise level, known kernel)
import csv
from . import config

def step3_params():
    # {sigma: (wiener_nsr, rl_iters)} from results/step3/summary.csv
    path = config.RESULTS_DIR / "step3" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -m experiments.step3_known_kernel first")
    with open(path) as f:
        return {float(r["sigma"]): (float(r["wiener_nsr"]), int(r["rl_iters"])) for r in csv.DictReader(f)}
