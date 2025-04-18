import json
import re
from pathlib import Path

import numpy as np

from AU_recognizer.core.util import logger, F_TAG, F_OUTPUT, P_PATH


def emoca_tag(diff_to_tag, threshold, project_data):
    logger.debug("Tagging vertices based on threshold...")
    project_name = str(project_data.sections()[0])
    output_path = Path(project_data[project_name][P_PATH]) / F_OUTPUT / F_TAG
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    tagged_vertices = [[] for _ in range(5023)]  # List of sequences for each vertex

    for diff_file in diff_to_tag:
        if diff_file.suffix.lower() == '.npy':
            code = extract_codes(diff_file.stem)  # Extract only the codes
            if not code:
                logger.warning(f"no code in file name {diff_file.stem}")
                continue  # Skip files with no valid code
            diffs = np.load(diff_file)  # Load the NumPy array
            if diffs.shape[0] != 5023:
                logger.warning(f"File {diff_file} has {diffs.shape[0]} vertices instead of 5023. Skipping.")
                continue  # Skip files with unexpected shapes

            # Normalize safely (avoid division by zero)
            min_val, max_val = np.min(diffs), np.max(diffs)
            if max_val == min_val:
                normalized_diffs = np.zeros_like(diffs)  # If all values are the same, set them to 0
            else:
                normalized_diffs = (diffs - min_val) / (max_val - min_val)

            for i, norm_diff in enumerate(normalized_diffs):
                if norm_diff >= threshold:
                    tagged_vertices[i].append(code)  # Append the tag code

    # Save tagged data to a file
    output_file = output_path / f"tagged_vertices_{threshold}.json"
    with open(output_file, "w") as f:
        json.dump(tagged_vertices, f, indent=2)  # Pretty-print for readability

    logger.info(f"Tagging completed. Results saved to {output_path}")


def extract_codes(filename):
    match = re.search(r"(\d+(?:_\d+)*)", filename)  # Find numbers separated by '_'
    return match.group(1) if match else ""  # Return the matched codes or empty string
