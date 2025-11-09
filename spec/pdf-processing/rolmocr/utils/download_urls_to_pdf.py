#!/usr/bin/env python3
"""
Download URLs and convert to PDFs.

Usage:
    python download_urls_to_pdf.py <input_file> <output_dir>
"""
import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def sanitize_filename(url):
    """Create safe filename from URL."""
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


def get_download_url(url):
    """Convert URL to downloadable format and categorize."""
    if 'arxiv.org/abs/' in url:
        pdf_url = url.replace('/abs/', '/pdf/')
        return 'arxiv', pdf_url if pdf_url.endswith('.pdf') else pdf_url + '.pdf'

    if url.endswith('.pdf') or '/pdf/' in url.lower():
        return 'pdf', url

    return 'web', url


def download_pdf(url, output_path, timeout=30):
    """Download PDF file."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.SSLError:
        # Retry with SSL verification disabled for servers with weak crypto
        try:
            logger.warning(f"SSL error, retrying without verification: {url}")
            response = requests.get(url, timeout=timeout, stream=True, verify=False)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def web_to_pdf(url, output_path, browser_context, timeout=30):
    """Convert web page to PDF using Playwright."""
    try:
        page = browser_context.new_page()
        page.goto(url, timeout=timeout * 1000, wait_until='load')

        # Wait a bit for JavaScript to render content (especially for Notion, SPA sites)
        page.wait_for_timeout(5000)

        page.pdf(
            path=str(output_path),
            format='A4',
            print_background=True,
            margin={'top': '1cm', 'right': '1cm', 'bottom': '1cm', 'left': '1cm'}
        )
        page.close()
        return True
    except Exception as e:
        logger.error(f"Failed to convert {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download URLs to PDFs")
    parser.add_argument('input_file', type=Path, help='File with URLs (one per line)')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Re-download existing files')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds for web pages (default: 30)')
    args = parser.parse_args()

    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    with open(args.input_file) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    logger.info(f"Found {len(urls)} URLs to process")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    url_mapping = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()

        for url in tqdm(urls, desc="Processing"):
            category, download_url = get_download_url(url)
            output_path = args.output_dir / sanitize_filename(url)
            filename = output_path.name

            if not args.force and output_path.exists():
                stats['skipped'] += 1
                url_mapping[filename] = url
                continue

            if category in ('pdf', 'arxiv'):
                success = download_pdf(download_url, output_path, args.timeout)
            else:
                success = web_to_pdf(url, output_path, context, args.timeout)

            if success:
                stats['success'] += 1
                url_mapping[filename] = url
            else:
                stats['failed'] += 1
                if output_path.exists():
                    output_path.unlink()

        context.close()
        browser.close()

    # Save URL mapping
    mapping_path = args.output_dir / 'url_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(url_mapping, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved URL mapping to {mapping_path}")

    logger.info(f"Complete - Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")


if __name__ == '__main__':
    main()
