import argparse
import importlib
import shutil
import time
import traceback

from experiments import step6_common as C

MODULES = {"6a": "experiments.step6a_pca_basis", "6b": "experiments.step6b_tune_dev",
           "6c": "experiments.step6c_test", "6d": "experiments.step6d_analyze"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=C.default_workers())
    ap.add_argument("--from", dest="start", choices=C.PHASES, default="6a")
    ap.add_argument("--redo", choices=C.PHASES)
    a = ap.parse_args()
    cfg, root = C.cfg_for(a.smoke), C.out_root(a.smoke)
    root.mkdir(parents=True, exist_ok=True)
    proto = C.config.RESULTS_DIR / "step6" / "PROTOCOL.txt"
    if a.smoke and proto.exists():
        shutil.copy(proto, root / "PROTOCOL.txt")
    log = C.get_logger(root, "queue")
    log.info(f"queue start: smoke={a.smoke} workers={a.workers} from={a.start} redo={a.redo}")
    if a.redo:
        (root / ".done" / a.redo).unlink(missing_ok=True)
    t_all = time.time()
    for ph in C.PHASES[C.PHASES.index(a.start):]:
        if C.is_done(root, ph):
            log.info(f"{ph}: already done, skipped")
            continue
        plog = C.get_logger(root, ph)
        log.info(f"{ph}: start")
        t0 = time.time()
        try:
            mod = importlib.import_module(MODULES[ph])
            kw = {"smoke": a.smoke} if ph in ("6c", "6d") else {}
            info = mod.run(cfg, root, a.workers, plog, **kw)
        except BaseException:
            msg = traceback.format_exc()
            plog.error(msg)
            log.error(f"{ph}: FAILED after {(time.time() - t0) / 60:.1f} min (see logs/{ph}.log); queue stopped")
            raise SystemExit(1)
        C.mark_done(root, ph, info or "")
        log.info(f"{ph}: done in {(time.time() - t0) / 60:.1f} min ({info})")
    log.info(f"queue finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
