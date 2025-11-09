#!/usr/bin/env python3
"""
Generate url_mapping.json from existing URLs file and PDFs.

Usage:
    python generate_url_mapping.py <urls_file> <pdf_dir>
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def sanitize_filename(url):
    """Create safe filename from URL - must match download_urls_to_pdf.py logic."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed = urlparse(url)
    path = parsed.path.strip('/')

    if path:
        segments = path.split('/')
        name = segments[-1] if segments[-1] else segments[-2] if len(segments) > 1 else 'page'
    else:
        name = parsed.netloc.replace('www.', '')

    name = re.sub(r'\.(html?|php|aspx?)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^\w\-_.]', '_', name)[:150]

    return f"{name}_{url_hash}.pdf"


def main():
    parser = argparse.ArgumentParser(description="Generate URL mapping from existing files")
    parser.add_argument('urls_file', type=Path, help='File with URLs (one per line)')
    parser.add_argument('pdf_dir', type=Path, help='Directory containing PDFs')
    args = parser.parse_args()

    if not args.urls_file.exists():
        print(f"Error: URLs file not found: {args.urls_file}")
        sys.exit(1)

    if not args.pdf_dir.exists():
        print(f"Error: PDF directory not found: {args.pdf_dir}")
        sys.exit(1)

    # Read URLs
    with open(args.urls_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Found {len(urls)} URLs in {args.urls_file}")

    # Generate mapping
    url_mapping = {}
    found = 0
    missing = 0

    for url in urls:
        filename = sanitize_filename(url)
        pdf_path = args.pdf_dir / filename

        if pdf_path.exists():
            url_mapping[filename] = url
            found += 1
        else:
            print(f"Warning: PDF not found for URL: {url}")
            print(f"  Expected: {filename}")
            missing += 1

    # Save mapping
    output_path = args.pdf_dir / 'url_mapping.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(url_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  Found: {found} PDFs")
    print(f"  Missing: {missing} PDFs")
    print(f"  Saved mapping to: {output_path}")


if __name__ == '__main__':
    main()
