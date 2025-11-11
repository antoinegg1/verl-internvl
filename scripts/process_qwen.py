import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def nearest_16(x: int) -> int:
    m = round(x / 16)
    if m < 1:
        m = 1
    return m * 16


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def process_one(args):
    img_path, input_dir, output_dir = args
    img = Image.open(img_path)
    w, h = img.size
    new_w = nearest_16(w)
    new_h = nearest_16(h)
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    rel_path = img_path.relative_to(input_dir)
    save_path = output_dir / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(save_path)
    return str(img_path)


def process_folder_parallel(input_dir: Path, output_dir: Path, max_workers: int = None) -> None:
    imgs = [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(p, input_dir, output_dir) for p in imgs]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(process_one, tasks), total=len(tasks), desc="Processing images"):
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resize images to nearest 16-multiple size (parallel).")
    parser.add_argument("input_dir", type=str, help="folder containing images")
    parser.add_argument("output_dir", type=str, help="folder to save processed images")
    parser.add_argument("--workers", type=int, default=64, help="number of processes (default: CPU count)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    process_folder_parallel(input_dir, output_dir, args.workers)


if __name__ == "__main__":
    main()
