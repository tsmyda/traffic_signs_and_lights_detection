from pathlib import Path

import numpy as np

from common import read_json


def load_training_samples(dataset_dir, settings, feature_count, max_positives, max_negatives, seed):
    paths = [dataset_dir / "samples" / f"{Path(value).stem}.npz" for value in settings["images"]]
    counts = []
    for index, path in enumerate(paths, start=1):
        report = read_json(path.with_suffix(".json"))
        counts.append((report["negative_samples"], report["positive_samples"]))
        if index % 10000 == 0:
            print(f"Indexed training samples: {index}/{len(paths)}", flush=True)
    available = np.asarray(counts, dtype=np.int64).sum(axis=0)
    limits = (max_negatives, max_positives)
    generator = np.random.default_rng(seed)
    selections = [np.sort(generator.choice(int(count), size=min(int(count), limit) if limit is not None else int(count), replace=False)) for count, limit in zip(available, limits)]
    total = sum(len(selection) for selection in selections)
    if any(len(selection) == 0 for selection in selections):
        raise ValueError("Both positive and negative samples are required")
    features = np.empty((total, feature_count), dtype=np.float32)
    labels = np.empty(total, dtype=np.uint8)
    offsets = [0, 0]
    output_offset = 0
    for index, (path, class_counts) in enumerate(zip(paths, counts), start=1):
        local_indices = []
        for label, count in enumerate(class_counts):
            selection = selections[label]
            begin, end = np.searchsorted(selection, [offsets[label], offsets[label] + count])
            local_indices.append(selection[begin:end] - offsets[label])
            offsets[label] += count
        if any(len(indices) for indices in local_indices):
            with np.load(path, allow_pickle=False) as data:
                source_labels = data["labels"]
                if tuple(int((source_labels == label).sum()) for label in (0, 1)) != class_counts:
                    raise ValueError(f"Sample counts changed: {path}")
                source_features = data["features"]
                for label, indices in enumerate(local_indices):
                    selected = np.flatnonzero(source_labels == label)[indices]
                    count = len(selected)
                    features[output_offset:output_offset + count] = source_features[selected]
                    labels[output_offset:output_offset + count] = label
                    output_offset += count
        if index % 10000 == 0:
            print(f"Loaded training samples: {index}/{len(paths)}", flush=True)
    if output_offset != total:
        raise ValueError("Selected sample counts do not match")
    report = {"seed": seed, "available_positive_samples": int(available[1]), "available_negative_samples": int(available[0]), "selected_positive_samples": len(selections[1]), "selected_negative_samples": len(selections[0]), "max_positive_samples": max_positives, "max_negative_samples": max_negatives, "feature_array_bytes": int(features.nbytes)}
    return features, labels, report
