"""
Prepare 8-class dataset (6 defect + Clean + Other) from source folders.
Run from project root: python scripts/prepare_8class_data.py --source_dir /path/to/defect
"""
import os
import shutil
import random
from pathlib import Path
import argparse

# 8 classes: Defect_1..Defect_6 + Clean + Other
CLASS_FOLDERS = {
    'bridge': 'bridge',           # Defect_1
    'cmp_scratch': 'cmp_scratch', # Defect_2
    'ler': 'ler',                 # Defect_3
    'opens': 'opens',             # Defect_4
    'crack': 'crack',             # Defect_5 (from via_crack)
    'short': 'short',             # Defect_6 (copy from bridge for 6th defect)
    'clean': 'clean',
    'other': 'other',
}

# Map source dataset folders to (defect_subfolder_name, class_name)
SOURCE_DATASETS = [
    ('bridge_clean_150_highquality', 'bridges', 'bridge'),
    ('cmp_clean_defect_150_highquality', 'cmp_scratch', 'cmp_scratch'),
    ('ler_clean_defect_150_highquality', 'ler', 'ler'),
    ('opens_clean_defect_150_highquality', 'opens', 'opens'),
    ('via_crack_clean_defect_150_highquality', 'crack', 'crack'),
]


def find_img_files(folder, suffix='.png'):
    """Find image files; handle both 'x.png' and 'x (1).png' naming."""
    if not folder.exists():
        return []
    files = []
    for f in folder.iterdir():
        if f.is_file() and (f.suffix.lower() == suffix or f.name.endswith(' (1).png')):
            files.append(f)
    return sorted(files, key=lambda x: x.name)


def copy_files(files, dst_dir, seed=42):
    """Copy files to dst_dir with normalized names."""
    if not files:
        return 0
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(sorted(files, key=lambda x: x.name)):
        shutil.copy2(f, dst_dir / f.name.replace(' (1)', '').replace(' ', '_'))
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Root dir containing e.g. bridge_clean_150_highquality, via_crack_clean_defect_150_highquality')
    parser.add_argument('--out_dir', type=str, default='data/raw',
                        help='Output root for 8-class folders')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    src = Path(args.source_dir)
    out = Path(args.out_dir)
    random.seed(args.seed)

    # 1) Defect_1..Defect_5 from source datasets
    clean_pool = []
    for folder_name, defect_sub, class_name in SOURCE_DATASETS:
        folder = src / folder_name
        if not folder.exists():
            # Try with " (1)" suffix (from zip extract)
            folder = src / (folder_name + ' (1)')
        if not folder.exists():
            print(f"Skip (not found): {folder_name}")
            continue
        defect_dir = folder / defect_sub
        if not defect_dir.exists():
            defect_dir = folder / (defect_sub + ' (1)')
        clean_dir = folder / 'clean'
        if not clean_dir.exists():
            clean_dir = folder / 'clean (1)'
        defect_imgs = find_img_files(defect_dir)
        clean_imgs = find_img_files(clean_dir)
        if defect_imgs:
            dst = out / class_name
            n = copy_files(defect_imgs[:75], dst, seed=args.seed)
            print(f"Defect {class_name}: {n} images")
        if clean_imgs:
            clean_pool.extend(clean_imgs)

    # 2) Defect_6 = short (copy from bridge defect images)
    bridge_defect = out / 'bridge'
    short_dir = out / 'short'
    if bridge_defect.exists():
        shorts = list(Path(bridge_defect).glob('*.png'))[:75]
        copy_files(shorts, short_dir, seed=args.seed)
        print(f"Defect short: {len(shorts)} images (from bridge)")

    # 3) Clean: merge up to 75 per source (or 375 total)
    clean_out = out / 'clean'
    clean_out.mkdir(parents=True, exist_ok=True)
    if clean_pool:
        chosen = random.sample(clean_pool, min(375, len(clean_pool)))
        for i, f in enumerate(chosen):
            shutil.copy2(f, clean_out / f"clean_{i+1}.png")
        print(f"Clean: {len(chosen)} images")

    # 4) Other: 50 images from defect sources (use images 76+ from each to avoid overlap)
    other_dir = out / 'other'
    other_pool = []
    for folder_name, defect_sub, _ in SOURCE_DATASETS:
        folder = src / folder_name
        if not folder.exists():
            folder = src / (folder_name + ' (1)')
        defect_dir = folder / defect_sub if (folder / defect_sub).exists() else folder / (defect_sub + ' (1)')
        imgs = find_img_files(defect_dir)
        if len(imgs) > 75:
            other_pool.extend(imgs[75:85])  # 10 per type
    other_dir.mkdir(parents=True, exist_ok=True)
    if len(other_pool) >= 50:
        chosen = random.sample(other_pool, 50)
        for i, f in enumerate(chosen):
            shutil.copy2(f, other_dir / f"other_{i+1}.png")
        print(f"Other: 50 images")
    else:
        # Fallback: copy from any defect folder
        for d in [out / c for c in ['bridge', 'cmp_scratch', 'ler', 'opens', 'crack'] if (out / c).exists()]:
            other_pool.extend(list(d.glob('*.png')))
        chosen = random.sample(other_pool, min(50, len(other_pool)))
        for i, f in enumerate(chosen):
            shutil.copy2(f, other_dir / f"other_{i+1}.png")
        print(f"Other: {len(chosen)} images")

    print(f"\n8-class data prepared under {out.absolute()}")


if __name__ == '__main__':
    main()
