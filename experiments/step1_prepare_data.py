from deblur import config
from deblur.data import collect_sources, save_sharp_set, sharpness
from deblur.plotting import chunk, image_grid
from deblur.tables import write_csv


def main():
    items = collect_sources()
    save_sharp_set(items)
    out = config.results_dir("step1")

    rows = [dict(name=n, shape=f"{im.shape[0]}x{im.shape[1]}", sharpness=round(sharpness(im), 5)) for n, im in items]
    write_csv(out / "sharpness.csv", rows)
    print(f"{'name':28s} {'shape':10s} {'sharpness':>10s}")
    for r in rows:
        print(f"{r['name']:28s} {r['shape']:10s} {r['sharpness']:10.5f}")

    panels = [(im, n, None) for n, im in items]
    image_grid(chunk(panels, 5), out / "contact_sheet.png", cell=3.0)
    print(f"\n{len(items)} images -> {config.SHARP_DIR}\nfigures -> {out}")


if __name__ == "__main__":
    main()
