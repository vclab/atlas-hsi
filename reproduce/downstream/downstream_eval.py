"""Downstream classification evaluation (ATLAS paper, Appendix G).

Per-pixel classification on the original cube, decoded reconstructions, and the
splatted abundance maps. Protocol: SVM-RBF (grid-searched C, gamma) plus random forest,
100 training pixels per class, 10 seeded draws (the paper's seed-paired samples),
OA/AA/kappa as mean and std. See README.md in this folder for the full command list.

Usage (from the repository root):
  python reproduce/downstream/downstream_eval.py --scene paviau --features original
  python reproduce/downstream/downstream_eval.py --scene paviau --features recon \
      --recon reproduce/downstream/recons/paviau/recon.npy

The "original" condition is the band-max-normalized cube from atlas.data.load_cube,
which is both what the model trains on and what it reconstructs; raw radiance is never
the comparison target.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from atlas.data import _load_envi, load_abundance, load_cube, rank_for  # noqa: E402

RAW = REPO / "data" / "raw"
RESULTS = HERE / "records"

SCENES = ("paviau", "paviac", "whu_honghu")


def _rel(path):
    """Record input paths relative to this folder so records carry no local layout."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(HERE))
    except ValueError:
        return p.name


def load_gt(scene):
    """GT raster cropped exactly as atlas.data.load_cube crops the cube."""
    if scene == "paviau":
        gt = scipy.io.loadmat(RAW / "PaviaU_gt.mat")["paviaU_gt"].astype(np.int64)
        return gt[-340:, :]
    if scene == "paviac":
        return scipy.io.loadmat(RAW / "Pavia_gt.mat")["pavia_gt"].astype(np.int64)
    if scene == "whu_honghu":
        return _load_envi(str(RAW / "WHU-Hi-HongHu_gt.bsq")).squeeze().astype(np.int64)
    raise ValueError(f"Unknown scene: {scene}")


def draw_split(y, n_per_class, rng):
    """Fixed-count stratified split. Returns (train_idx, test_idx) into y.

    For each class present in y, sample a fixed number of training pixels without
    replacement; every remaining labeled pixel is test. Requirements:
      - train and test are disjoint; together they cover all of y
      - a class smaller than 2 * n_per_class must still keep test pixels, so cap
        its training count (e.g. at half its size) rather than swallowing it
      - deterministic given rng (a numpy Generator); use rng, never np.random.*
    """

    train_idx = np.array([], dtype=np.int64)
    test_idx = np.array([], dtype=np.int64)

    for i in np.unique(y):
        idx = np.where(y == i)[0]

        shuffled_arr = rng.permutation(idx)

        k = min(n_per_class, len(idx) // 2)
        train = shuffled_arr[:k]
        test = shuffled_arr[k:]

        train_idx = np.concat((train_idx, train))
        test_idx = np.concat((test_idx, test))

    return (train_idx, test_idx)


def draw_split_blocked(y, block_ids, n_per_class, rng, train_frac=0.5):
    """Spatially blocked split: whole blocks go to the train or test pool, training
    pixels are sampled only from train-pool blocks, and the test set is every labeled
    pixel in test-pool blocks. Train-pool pixels beyond the per-class draw are
    discarded rather than tested: they are spatially adjacent to training pixels,
    which is exactly the channel codec smoothing leaks class labels through. Blocks
    have no guard band, so leakage across block borders is greatly reduced rather
    than eliminated (quantify the residual by raising --block-size).

    The block assignment is redrawn (seeded, up to 20 tries) until every class has
    pixels on both sides; a class's train count is capped by what its train blocks
    hold, so rare localized classes may train on fewer than n_per_class pixels.
    """
    blocks = np.unique(block_ids)
    n_tr = max(1, int(len(blocks) * train_frac))
    classes = np.unique(y)
    for _attempt in range(20):
        tr_blocks = rng.permutation(blocks)[:n_tr]
        in_tr = np.isin(block_ids, tr_blocks)
        if all(((y == c) & in_tr).any() and ((y == c) & ~in_tr).any() for c in classes):
            break
    else:
        raise RuntimeError(
            "blocked split: no block assignment covered every class on both sides in "
            "20 tries; use smaller blocks or drop the scene from the blocked protocol")
    train_parts = []
    for c in classes:
        idx = np.where((y == c) & in_tr)[0]
        train_parts.append(rng.permutation(idx)[:min(n_per_class, len(idx))])
    return np.concat(train_parts), np.where(~in_tr)[0]


def fit_predict_svm(Xtr, ytr, Xte):
    """Grid-searched SVM-RBF. Scaler and grid search see training pixels only."""
    scaler = StandardScaler().fit(Xtr)
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1]},
        cv=3,
        n_jobs=-1,
    )
    grid.fit(scaler.transform(Xtr), ytr)
    return grid.best_estimator_.predict(scaler.transform(Xte)), grid.best_params_


def fit_predict_rf(Xtr, ytr, Xte, seed):
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte), {"n_estimators": 200}


def score(y_true, y_pred):
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "aa": float(recall_score(y_true, y_pred, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--features", required=True,
                    choices=["original", "recon", "abundance", "a0"],
                    help="original: normalized cube; recon: cube from --recon; "
                         "abundance: splatted K-dim map from --abundance (compressed "
                         "domain, no cube materialized); a0: NMF abundances of the "
                         "original cube from data/init (rank-reduction control)")
    ap.add_argument("--recon", type=Path,
                    help="(H, W, B) .npy reconstruction, required for --features recon")
    ap.add_argument("--abundance", type=Path,
                    help="(H, W, K) .npy splatted abundance map, required for "
                         "--features abundance (codec.decompress(path, abundance=True))")
    ap.add_argument("--rank", type=int, default=None,
                    help="NMF rank for --features a0 (default: the scene's standard rank)")
    ap.add_argument("--split", choices=["random", "blocked"], default="random",
                    help="blocked: train and test come from disjoint 2D blocks, "
                         "greatly reducing label leakage through codec spatial smoothing")
    ap.add_argument("--block-size", type=int, default=32,
                    help="block edge in pixels for --split blocked")
    ap.add_argument("--train-per-class", type=int, default=100)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    args = ap.parse_args()

    gt = load_gt(args.scene)
    if args.features == "original":
        feats = load_cube(args.scene, data_dir=str(RAW))
    elif args.features == "recon":
        if args.recon is None:
            ap.error("--features recon requires --recon")
        feats = np.load(args.recon)
    elif args.features == "abundance":
        if args.abundance is None:
            ap.error("--features abundance requires --abundance")
        feats = np.load(args.abundance)
    else:  # a0: the control pricing rank reduction alone (same K, no codec involved)
        rank = args.rank or rank_for(args.scene)
        a0 = load_abundance(args.scene, rank, init_dir=str(REPO / "data" / "init"))
        if a0 is None:
            ap.error(f"no NMF init for {args.scene} at rank {rank} in data/init/")
        feats = a0.reshape(gt.shape[0], gt.shape[1], rank)
    assert feats.shape[:2] == gt.shape, f"cube {feats.shape[:2]} vs GT {gt.shape}"

    mask = gt > 0
    X = feats[mask].astype(np.float32)
    y = gt[mask]
    # np.where yields row-major coordinates, the same order boolean masking flattens
    # X and y in, so block_ids[i] is the block of pixel X[i].
    rows, cols = np.where(mask)
    block_ids = (rows // args.block_size) * (gt.shape[1] // args.block_size + 1) \
        + (cols // args.block_size)
    print(f"{args.scene}/{args.features}: {X.shape[0]} labeled px, "
          f"{X.shape[1]}-dim features, {len(np.unique(y))} classes, "
          f"split={args.split}")

    per_draw = []
    for d in range(args.draws):
        rng = np.random.default_rng(args.seed + d)
        if args.split == "blocked":
            tr, te = draw_split_blocked(y, block_ids, args.train_per_class, rng)
        else:
            tr, te = draw_split(y, args.train_per_class, rng)
        t0 = time.time()
        svm_pred, svm_params = fit_predict_svm(X[tr], y[tr], X[te])
        rf_pred, rf_params = fit_predict_rf(X[tr], y[tr], X[te], seed=args.seed + d)
        row = {
            "draw": d,
            "svm": score(y[te], svm_pred) | {"params": svm_params},
            "rf": score(y[te], rf_pred) | {"params": rf_params},
        }
        per_draw.append(row)
        print(f"  draw {d}: SVM OA {row['svm']['oa']:.4f} | RF OA {row['rf']['oa']:.4f}"
              f"  ({time.time() - t0:.0f}s)")

    summary = {}
    for clf in ("svm", "rf"):
        for metric in ("oa", "aa", "kappa"):
            vals = [r[clf][metric] for r in per_draw]
            summary[f"{clf}_{metric}"] = {"mean": float(np.mean(vals)),
                                          "std": float(np.std(vals))}

    RESULTS.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    if args.split == "blocked":
        tag = f"_blocked{tag}"
    out = RESULTS / f"{args.scene}_{args.features}{tag}.json"
    out.write_text(json.dumps({
        "scene": args.scene, "features": args.features,
        "recon": _rel(args.recon) if args.recon else None,
        "split": args.split, "block_size": args.block_size,
        "train_per_class": args.train_per_class, "draws": args.draws,
        "seed": args.seed, "summary": summary, "per_draw": per_draw,
    }, indent=2))

    print(f"\n{'':12s} {'OA':>16s} {'AA':>16s} {'kappa':>16s}")
    for clf in ("svm", "rf"):
        cells = [f"{summary[f'{clf}_{m}']['mean']:.4f} ± {summary[f'{clf}_{m}']['std']:.4f}"
                 for m in ("oa", "aa", "kappa")]
        print(f"{clf.upper():12s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
