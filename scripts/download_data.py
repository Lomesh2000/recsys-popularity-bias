"""Download datasets for experiments."""

import os
import argparse
import urllib.request
import zipfile
from typing import Optional


def download_file(url: str, dest_path: str) -> None:
    """Download file from URL with progress."""
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 100 == 0 or percent >= 100:
            print(f"  Progress: {percent}%", end="\r")

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print(f"\nSaved to {dest_path}")


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete")


def download_movielens(variant: str = '1m', data_dir: str = './data') -> None:
    """Download MovieLens dataset."""
    urls = {
        '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    }

    if variant not in urls:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(urls.keys())}")

    url = urls[variant]
    dest = os.path.join(data_dir, f'ml-{variant}.zip')

    if os.path.exists(dest):
        print(f"File already exists: {dest}")
        return

    download_file(url, dest)
    extract_zip(dest, data_dir)
    print(f"MovieLens-{variant} ready at {os.path.join(data_dir, f'ml-{variant}')}")


def main():
    parser = argparse.ArgumentParser(description='Download recommendation datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'movielens-1m', 'movielens-100k', 'goodreads', 'google-reviews'],
                       help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to save datasets')
    args = parser.parse_args()

    if args.dataset in ('all', 'movielens-1m'):
        download_movielens('1m', args.data_dir)
    if args.dataset in ('all', 'movielens-100k'):
        download_movielens('100k', args.data_dir)

    if args.dataset in ('all', 'goodreads', 'google-reviews'):
        print("\nNote: Goodreads and Google Reviews require manual download.")
        print("  Goodreads: https://github.com/BahramJannesar/GoodreadsBookDataset")
        print("  Google Reviews: https://jiachengli1995.github.io/google/index.html")
        print(f"  Place files in {args.data_dir}/goodreads/ and {args.data_dir}/google-reviews/")

    print("\nDownload complete!")


if __name__ == '__main__':
    main()
