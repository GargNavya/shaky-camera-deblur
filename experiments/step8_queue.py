import argparse
import importlib
import shutil
import time
import traceback

from experiments import step8_common as C

MODULES = {"8tune": "experiments.step8_tune", "8test": "experiments.step8_test",
           "8analyze": "experiments.step8_analyze", "8real": "experiments.step8_real"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=C.default_workers())
    ap.add_argument("--only", choices=C.PHASES, help="run just this phase (even if done before)")
    a = ap.parse_args()
    cfg, root = C.cfg_for(a.smoke), C.out_root(a.smoke)
    root.mkdir(parents=True, exist_ok=True)
    proto = C.config.RESULTS_DIR / "step8" / "PROTOCOL.txt"
    if a.smoke and proto.exists():
        shutil.copy(proto, root / "PROTOCOL.txt")
    log = C.get_logger(root, "queue")
    log.info(f"queue start: smoke={a.smoke} workers={a.workers} only={a.only}")
    t_all = time.time()
    for ph in ([a.only] if a.only else C.PHASES):
        if C.is_done(root, ph) and not a.only:
            log.info(f"{ph}: already done, skipped")
            continue
        if a.only in ("8tune", "8test") and C.is_done(root, ph):
            log.info(f"{ph}: already done; its cases are reused (nothing is recomputed or overwritten)")
        plog = C.get_logger(root, ph)
        log.info(f"{ph}: start")
        t0 = time.time()
        try:
            info = importlib.import_module(MODULES[ph]).run(cfg, root, a.workers, plog, smoke=a.smoke)
        except BaseException:
            plog.error(traceback.format_exc())
            log.error(f"{ph}: FAILED after {(time.time() - t0) / 60:.1f} min (see logs/{ph}.log); queue stopped")
            raise SystemExit(1)
        C.mark_done(root, ph, info or "")
        log.info(f"{ph}: done in {(time.time() - t0) / 60:.1f} min ({info})")
    log.info(f"queue finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
