import argparse
import os
import time

import numpy as np

from .data import DEFAULT_DATA_DIR, DEFAULT_INIT_DIR, load_cube


def nmf_initialization(I, rank, dataset_name, init_dir=DEFAULT_INIT_DIR):
    """Run NMF on a (channels, H*W) matrix and save endmember/abundance .npy files."""
    from sklearn.decomposition import NMF  # optional heavy dep, imported lazily

    print(f"Running NMF initialization on {dataset_name} with rank={rank}")
    nmf = NMF(rank, init='random', random_state=42, max_iter=12000)
    endmember = nmf.fit_transform(I).T  # shape (rank, channels)
    abundance = nmf.components_.T       # shape (H*W, rank)

    os.makedirs(init_dir, exist_ok=True)
    np.save(os.path.join(init_dir, f"{dataset_name}_endmember_rank_{rank}_NMF.npy"), endmember)
    np.save(os.path.join(init_dir, f"{dataset_name}_abundance_rank_{rank}_NMF.npy"), abundance)
    print(f"Saved NMF initialization results to {init_dir}.")


def main():
    parser = argparse.ArgumentParser(description="NMF initialization for hyperspectral endmembers.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. paviau) or a .mat/.npy cube path")
    parser.add_argument("--rank", type=int, default=12, help="Number of endmembers (rank)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory holding raw dataset files")
    parser.add_argument("--init-dir", type=str, default=DEFAULT_INIT_DIR,
                        help="Directory to write NMF .npy outputs")
    args = parser.parse_args()

    start_time = time.time()
    cube = load_cube(args.dataset, data_dir=args.data_dir)        # (H, W, C), normalized
    I = np.transpose(cube, (2, 0, 1)).reshape(cube.shape[2], -1)  # (C, H*W)
    name = os.path.splitext(os.path.basename(args.dataset))[0].lower()
    nmf_initialization(I, args.rank, name, init_dir=args.init_dir)
    print(f"Finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
