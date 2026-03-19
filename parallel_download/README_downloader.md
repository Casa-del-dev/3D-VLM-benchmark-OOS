# Parallel HD-EPIC Downloader (Participant-Focused)

This folder contains a parallel downloader that mirrors the official `hd-epic-downloader` file organization.

## What it downloads

- Participant-scoped categories for the selected participant:
  - `Videos/P0X/...`
  - `SLAM-and-Gaze/P0X/...`
  - `Audio-HDF5/P0X/...`
- Global category included by default:
  - `Digital-Twin/...`
- Root file(s), e.g. `readme.txt`

## What it excludes

- `Hands-Masks/...` (global and not participant-specific)
- `VRS/...` by default (enable with `--include-vrs`)

## Output location

The script writes to `./HD-EPIC` by default (current working directory).

You can choose the parent directory with `--output-path`.
Example: `--output-path /tmp/data` writes to `/tmp/data/HD-EPIC`.

## Run

```bash
python parallel_download/downloader.py
```

Pick a participant:

```bash
python parallel_download/downloader.py --participant P02
```

Include VRS too:

```bash
python parallel_download/downloader.py --participant P02 --include-vrs
```

Optional tuning:

```bash
python parallel_download/downloader.py --workers 16 --retries 3 --timeout-sec 180
```

Custom parent output directory (still writes to `HD-EPIC` inside it):

```bash
python parallel_download/downloader.py --participant P02 --output-path /tmp/data
```

**Aria2c-style segmented downloading for large files**:

```bash
python parallel_download/downloader.py --participant P02 --workers 16 --segments-per-file 8 --split-threshold-mib 256
```



Notes:
- `--workers` controls parallel files.
- `--segments-per-file` controls parallel chunks within each eligible file.
- Segmentation is used only when the server supports HTTP range requests and file size is above `--split-threshold-mib`.

## Progress bars

The script shows separate tqdm bars per data category (`videos`, `slam-and-gaze`, `audio-hdf5`, `digital-twin`, etc.) and one file-level completion bar.
Each category bar displays transfer speed and downloaded bytes.
