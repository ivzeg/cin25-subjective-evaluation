# Subjective Evaluation of Image Colorization Quality Using the CIN25 Dataset
The repository contains the official data processing and statistical analysis pipeline for the paper Subjective Evaluation of Image Colorization Quality Using the CIN25 Dataset accepted for ICIP 2026 Workshop.

## Input Data Structure
Before running the script, ensure the file paths in the script point to your local data.

### 1. Subjective Data
Results_raw_all.csv: The raw subjective evaluation log. The zip file which contains CIN25 images, raw subjective data and MOS can be found at the webpage of our laboratory: [https://www.vcl.fer.hr/quality/cin25.html](https://www.vcl.fer.hr/quality/cin25.html).

Expected columns (no header): scene_id, image_path, method, score, user_id.

### 2. Objective Metric Data
The script expects pre-computed objective IQA metrics. For the paper, metrics were calculated using pyiqa framework: [IQA-PyTorch]([url](https://github.com/chaofengc/iqa-pytorch)).

cin25_psnr_rgb.csv

cin25_ssim_rgb.csv

cin25_lpips.csv

cin25_colorfulness.csv

Expected columns: image_name, metric_value.

## Usage
Simply execute the main script from your terminal or command prompt:

`python analysis_subjective_experiment_CIN25.py`

--------------------------------------------------------------------------------
## Citation

If you use this repository, please cite our work:

```bibtex
@inproceedings{zeger2026cin25,
  title={Subjective Evaluation of Image Colorization Quality Using the CIN25 Dataset},
  author={Ivana Zeger, Jan Muric, Patrik Blaskovic, Ivan Setka and Sonja Grgic},
  booktitle={Proceedings of the IEEE International Conference on Image Processing (ICIP)},
  pages={xxx--xxx},
  year={2026},
  organization={IEEE},
  doi={10.1109/ICIPxxxxxx.2026.xxxxxxx}
}
```

--------------------------------------------------------------------------------
## Acknowledgments

This work utilizes the following open-source project:

- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), which provides implementations of multiple image quality assessment metrics, including PSNR, SSIM, LPIPS, and colorfulness.

We thank the authors for making their code and models publicly available.
