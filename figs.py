#!/usr/bin/env python3

import os
import glob
import sys
import subprocess
import argparse


def search_html_files(keywords: list[str], figures_dir: str = None) -> list[str]:
    """
    Search for HTML files in the figures directory that contain all given keywords.
    
    Args:
        keywords: List of keywords to search for (all must be present)
        figures_dir: Directory to search in (defaults to figures/ relative to this file)
    
    Returns:
        List of full paths to HTML files matching all keywords
    """
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    
    if not os.path.exists(figures_dir):
        return []
    
    # Get all HTML files in the directory
    html_files = glob.glob(os.path.join(figures_dir, "*.html"))
    
    # Filter files that contain all keywords (case-insensitive)
    matching_files = []
    
    for file_path in html_files:
        filename = os.path.basename(file_path).lower()
        if all(keyword.lower() in filename for keyword in keywords):
            matching_files.append(file_path)
    
    return sorted(matching_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for HTML files in figures/ directory containing all keywords."
    )
    parser.add_argument(
        "keywords",
        nargs="+",
        help="Keywords to search for (all must be present in filename)"
    )
    parser.add_argument(
        "-o", "--open",
        action="store_true",
        help="Open all found files with Chrome"
    )
    
    args = parser.parse_args()
    
    results = search_html_files(args.keywords)
    
    if results:
        for file_path in results:
            print(file_path)
        
        if args.open:
            # Try to open files with Chrome
            chrome_commands = [
                "google-chrome",
                "google-chrome-stable",
                "google-chrome-unstable",
                "chrome",
            ]
            
            chrome_cmd = None
            for cmd in chrome_commands:
                try:
                    subprocess.run(
                        [cmd, "--version"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                    chrome_cmd = cmd
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if chrome_cmd is None:
                print("Error: Chrome not found. Please install Chrome or Chromium.", file=sys.stderr)
                sys.exit(1)
            
            # Open all files with Chrome (detached so script can exit)
            subprocess.Popen(
                [chrome_cmd] + results,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
    else:
        print(f"No HTML files found containing all keywords: {', '.join(args.keywords)}", file=sys.stderr)
        sys.exit(1)

