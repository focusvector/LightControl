"""
Create a cleaned dataset with only complete samples.
Each sample must have: RGB, mask, shadow, and metadata files.
"""

import os
import shutil
import json
from pathlib import Path
from glob import glob

SOURCE_DIR = "/home/razz/Downloads/output"
TARGET_DIR = "/home/razz/Downloads/output_cleaned"

def validate_sample(idx, source_dir):
    """
    Check if a sample has ALL required files and valid metadata.
    
    Required files:
    - rgb_{idx}.png
    - mask_{idx}.png
    - shadow_{idx}.png
    - meta_{idx}.json
    
    Returns:
        (is_valid, reason) tuple
    """
    files = {
        'rgb': os.path.join(source_dir, f'rgb_{idx:05d}.png'),
        'mask': os.path.join(source_dir, f'mask_{idx:05d}.png'),
        'shadow': os.path.join(source_dir, f'shadow_{idx:05d}.png'),
        'meta': os.path.join(source_dir, f'meta_{idx:05d}.json')
    }
    
    # Check all files exist
    for file_type, path in files.items():
        if not os.path.exists(path):
            return False, f"Missing {file_type}"
        if os.path.getsize(path) == 0:
            return False, f"Empty {file_type}"
    
    # Validate metadata
    try:
        with open(files['meta'], 'r') as f:
            meta = json.load(f)
        
        # Check required fields
        required_fields = ['theta', 'phi', 'size']
        for field in required_fields:
            if field not in meta:
                return False, f"Missing metadata field: {field}"
            
            # Check valid ranges
            if field == 'theta' and not (0 <= meta[field] <= 90):
                return False, f"Invalid theta: {meta[field]}"
            if field == 'phi' and not (0 <= meta[field] <= 360):
                return False, f"Invalid phi: {meta[field]}"
            if field == 'size' and not (0.1 <= meta[field] <= 10.0):
                return False, f"Invalid size: {meta[field]}"
    
    except json.JSONDecodeError:
        return False, "Invalid JSON"
    except Exception as e:
        return False, f"Metadata error: {str(e)}"
    
    return True, "Valid"

def copy_sample(idx, source_dir, target_dir):
    """Copy all files for a valid sample."""
    files = [
        f'rgb_{idx:05d}.png',
        f'mask_{idx:05d}.png',
        f'shadow_{idx:05d}.png',
        f'meta_{idx:05d}.json'
    ]
    
    for filename in files:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)
        shutil.copy2(src, dst)

def main():
    print("=" * 70)
    print("DATASET CLEANING: Keep only complete samples")
    print("=" * 70)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    
    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Find all RGB files (use as base for sample discovery)
    rgb_files = sorted(glob(os.path.join(SOURCE_DIR, "rgb_*.png")))
    total_samples = len(rgb_files)
    
    if total_samples == 0:
        print(f"\n❌ No samples found in {SOURCE_DIR}")
        return
    
    print(f"\nScanning {total_samples} samples...")
    print("-" * 70)
    
    valid_count = 0
    invalid_count = 0
    invalid_reasons = {}
    valid_indices = []
    
    for rgb_file in rgb_files:
        # Extract sample index from filename
        idx = int(os.path.basename(rgb_file).split('_')[1].split('.')[0])
        
        is_valid, reason = validate_sample(idx, SOURCE_DIR)
        
        if is_valid:
            copy_sample(idx, SOURCE_DIR, TARGET_DIR)
            valid_count += 1
            valid_indices.append(idx)
            
            if valid_count <= 10:  # Show first 10
                print(f"  ✓ Sample {idx:05d}: Complete")
        else:
            invalid_count += 1
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            
            if invalid_count <= 10:  # Show first 10
                print(f"  ❌ Sample {idx:05d}: {reason}")
    
    print("-" * 70)
    print(f"\nRESULTS:")
    print(f"  Total samples scanned: {total_samples}")
    print(f"  Valid samples:         {valid_count} ({100*valid_count/total_samples:.1f}%)")
    print(f"  Invalid samples:       {invalid_count} ({100*invalid_count/total_samples:.1f}%)")
    
    if invalid_reasons:
        print(f"\nInvalid sample breakdown:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            print(f"  • {reason}: {count}")
    
    print(f"\n✓ Cleaned dataset saved to: {TARGET_DIR}")
    print("=" * 70)
    
    # Save cleaning report
    report = {
        'source_dir': SOURCE_DIR,
        'target_dir': TARGET_DIR,
        'total_samples': total_samples,
        'valid_samples': valid_count,
        'invalid_samples': invalid_count,
        'invalid_reasons': invalid_reasons,
        'valid_indices': valid_indices[:100]  # First 100 valid indices
    }
    
    report_path = os.path.join(TARGET_DIR, "cleaning_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Cleaning report saved to: {report_path}\n")

if __name__ == '__main__':
    main()
