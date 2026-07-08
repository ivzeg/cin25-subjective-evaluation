#!/usr/bin/env python3

import csv
import os
import math
from collections import defaultdict
from statistics import median, stdev

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --- PATHS AND PARAMETERS ---

# go to the bottom of the following page to download the zip which contains CIN25 images, raw subjective data and MOS:
# https://www.vcl.fer.hr/quality/cin25.html

CSV_PATH = r"D:\CIN25\Results_raw_all.csv" # change the path
OUTPUT_CSV = "MOS_subj_experiment.csv"

# the following files are obtained using implementations of metrics in pyiqa framework
# https://github.com/chaofengc/iqa-pytorch

IQA_FILES = {
    "PSNR": r".\cin25_psnr_rgb.csv",
    "SSIM": r".\cin25_ssim_rgb.csv",
    "LPIPS": r".\cin25_lpips.csv",
    "COLORFULNESS": r".\cin25_colorfulness.csv"
}

ROBUST_Z_THRESHOLD = 3.5
IDENTICAL_DIFF_THRESHOLD = 2
BAD_GAN_IMAGES = {"860_gan", "114_gan", "232_gan", "440_gan", "814_gan"}
IDENTICAL_PAIR = ("979_cnn", "979_tf")
COLOR_METHODS = {"cnn", "gan", "tf"}

CATEGORIES = {
    "Indoor": ["050", "403"],
    "Outdoor": ["452", "522", "611", "898", "979"],
    "Urban": ["067", "114", "419"],
    "Humans": ["088", "232", "384", "399", "440", "814", "860"],
    "Flora & Fauna": ["007", "040", "104", "168", "273", "794"],
    "Challenging": ["005", "351"]
}
SCENE_TO_CAT = {scene: cat for cat, scenes in CATEGORIES.items() for scene in scenes}


# --- HELPERS ---
def robust_z(score, med, mad):
    return 0.6745 * (score - med) / mad if mad != 0 else 0.0

def get_linear_rmse(x, y):
    n = len(x)
    if n < 2: return 0.0
    sum_x, sum_y = sum(x), sum(y)
    sum_xx = sum(xi ** 2 for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    denom = (n * sum_xx - sum_x ** 2)
    if denom == 0:
        m, c = 0.0, sum_y / n
    else:
        m = (n * sum_xy - sum_x * sum_y) / denom
        c = (sum_y - m * sum_x) / n
    mse = sum((yi - (m * xi + c)) ** 2 for xi, yi in zip(x, y)) / n
    return math.sqrt(mse)

def min_max_normalize(data_dict, invert=False):
    if not data_dict: return {}
    vals = list(data_dict.values())
    v_min, v_max = min(vals), max(vals)
    norm_dict = {}
    for k, v in data_dict.items():
        val = (v - v_min) / (v_max - v_min) if v_max > v_min else 0.0
        norm_dict[k] = 1.0 - val if invert else val
    return norm_dict

def fmt_p(p):
    return "<.001" if p < 0.001 else f"={p:.3f}"

# --- LOAD DATA AND SCREEN PARTICIPANTS ---

raw_rows = []
scores_per_image = defaultdict(list)
user_all_scores = defaultdict(list)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        scene_id, img_path, method, score, user_id = row
        img_name = os.path.basename(img_path)
        val = float(score)

        raw_rows.append({
            "scene": scene_id, "image": img_name,
            "method": method.lower(), "score": val, "user": user_id
        })
        scores_per_image[img_name].append(val)
        user_all_scores[user_id].append(val)

image_stats = {img: (median(scores), median([abs(s - median(scores)) for s in scores]))
               for img, scores in scores_per_image.items()}

user_flags = defaultdict(int)
user_data_map = defaultdict(lambda: defaultdict(dict))
trap_counts = {"Z-Score Outlier": 0, "Semantic (Bad GAN)": 0, "Identical Mismatch": 0}

for r in raw_rows:
    u, img, score, method, scene = r["user"], r["image"], r["score"], r["method"], r["scene"]
    user_data_map[u][scene][method] = (score, img)

    # trap 1 (outliers) and trap 2 (semantic attentiveness)
    med, mad = image_stats[img]
    if abs(robust_z(score, med, mad)) > ROBUST_Z_THRESHOLD:
        user_flags[u] += 1
        trap_counts["Z-Score Outlier"] += 1
    if img.split(".")[0] in BAD_GAN_IMAGES and score > 3:
        user_flags[u] += 1
        trap_counts["Semantic (Bad GAN)"] += 1

# trap 3 (consistency check)
for u, scenes in user_data_map.items():
    flat_scores = {img.split(".")[0]: score for s in scenes.values() for method, (score, img) in s.items()}
    if all(k in flat_scores for k in IDENTICAL_PAIR):
        if abs(flat_scores[IDENTICAL_PAIR[0]] - flat_scores[IDENTICAL_PAIR[1]]) >= IDENTICAL_DIFF_THRESHOLD:
            user_flags[u] += 1
            trap_counts["Identical Mismatch"] += 1

valid_users = [u for u in user_all_scores.keys() if user_flags[u] <= 1]
excluded_users = [u for u in user_all_scores.keys() if user_flags[u] > 1]

print("\n PARTICIPANT SCREENING \n")
print(f"Total Users: {len(user_all_scores)} | Valid for MOS: {len(valid_users)} | Excluded: {len(excluded_users)}")
if excluded_users:
    print(f"  Excluded User IDs: {', '.join(map(str, sorted(excluded_users)))}")
else:
    print("  Excluded User IDs: None")
for k, v in trap_counts.items(): print(f"  Total Flagged Occurrences ({k}): {v}")

drift_diffs = []
for u in valid_users:
    scenes = user_data_map[u]
    flat_scores = {img.split(".")[0]: score for s in scenes.values() for method, (score, img) in s.items()}
    if all(k in flat_scores for k in IDENTICAL_PAIR):
        drift_diffs.append(abs(flat_scores[IDENTICAL_PAIR[0]] - flat_scores[IDENTICAL_PAIR[1]]))
print(
    f"Human Noise Floor (Scene 979 average variance): {sum(drift_diffs) / len(drift_diffs) if drift_diffs else 0:.3f} points\n")

# --- Z-SCORE NORMALIZATION AND FINAL MOS ---

user_norm_params = {}
for u in valid_users:
    scores = user_all_scores[u]
    u_mean, u_std = sum(scores) / len(scores), stdev(scores) if len(scores) > 1 else 0.1
    user_norm_params[u] = (u_mean, u_std if u_std != 0 else 0.1)

final_mos_collector = defaultdict(list)
final_likert_collector = defaultdict(list)

for r in raw_rows:
    if r["user"] in valid_users and r["method"] in COLOR_METHODS:
        u_mean, u_std = user_norm_params[r["user"]]
        final_mos_collector[r["image"]].append((r["score"] - u_mean) / u_std)
        final_likert_collector[r["image"]].append(r["score"])

mos_results = {img: sum(zs) / len(zs) for img, zs in final_mos_collector.items()}
likert_mos_results = {img: sum(scs) / len(scs) for img, scs in final_likert_collector.items() if scs}

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for img_name in sorted(mos_results.keys()):
        writer.writerow([img_name, f"{mos_results[img_name]:.6f}"])

# --- CORRELATION ACROSS SEMANTIC CATEGORIES AND ACROSS ALL COLORIZED IMAGES ---

print(" METRIC BENCHMARKING (PEARSON, SPEARMAN, RMSE) ")
mos_by_cat = defaultdict(dict)
for img, val in mos_results.items():
    scene = img.split("_")[0]
    if scene in SCENE_TO_CAT:
        mos_by_cat[SCENE_TO_CAT[scene]][img] = val

benchmark_data = defaultdict(dict)
all_iqa_raw = defaultdict(dict)

for metric_name, path in IQA_FILES.items():
    if not os.path.exists(path): continue
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            img = row[0].strip()
            method = img.split("_")[-1].split(".")[0].lower()
            if method in COLOR_METHODS:
                all_iqa_raw[metric_name][img] = float(row[1])

    for cat, mos_dict in mos_by_cat.items():
        common = sorted(set(mos_dict.keys()) & set(all_iqa_raw[metric_name].keys()))
        if len(common) < 2: continue
        m_vals, q_vals = [mos_dict[i] for i in common], [all_iqa_raw[metric_name][i] for i in common]
        l_vals = [likert_mos_results.get(i, 0.0) for i in common]

        pr, pp = pearsonr(m_vals, q_vals)
        sr, sp = spearmanr(m_vals, q_vals)
        rmse = get_linear_rmse(q_vals, l_vals)
        benchmark_data[metric_name][cat] = (pr, pp, sr, sp, rmse)

    # global (all colorized images)
    global_common = sorted(set(mos_results.keys()) & set(all_iqa_raw[metric_name].keys()))
    if len(global_common) >= 2:
        m_vals, q_vals = [mos_results[i] for i in global_common], [all_iqa_raw[metric_name][i] for i in global_common]
        l_vals = [likert_mos_results.get(i, 0.0) for i in global_common]

        pr, pp = pearsonr(m_vals, q_vals)
        sr, sp = spearmanr(m_vals, q_vals)
        rmse = get_linear_rmse(q_vals, l_vals)
        benchmark_data[metric_name]["Global"] = (pr, pp, sr, sp, rmse)

for metric_name in benchmark_data.keys():
    print(f"\n--- {metric_name} ---")
    print(f"{'Category':<15} | {'Pearson (r)':<18} | {'Spearman (ρ)':<18} | {'RMSE':<8}")
    print("-" * 65)
    for cat in list(CATEGORIES.keys()) + ["Global"]:
        if cat in benchmark_data[metric_name]:
            pr, pp, sr, sp, rmse = benchmark_data[metric_name][cat]
            p_str = f"r={pr:>6.3f} (p{fmt_p(pp):<5})"
            s_str = f"ρ={sr:>6.3f} (p{fmt_p(sp):<5})"
            print(f"{cat:<15} | {p_str:<18} | {s_str:<18} | {rmse:>5.3f}")

# --- EXTREMES OF AGREEMENT ---

print("\n EXTREMES OF AGREEMENT \n")
norm_mos = min_max_normalize(mos_results)

print(f"{'Metric':<10} | {'Match':<5} | {'Image':<10} | {'Norm. MOS':<8} | {'Norm. IQA':<8} | {'Error (delta)':<8}")
print("-" * 65)

for metric_name in benchmark_data.keys():
    is_dist = (metric_name == "LPIPS")
    norm_iqa = min_max_normalize(all_iqa_raw[metric_name], invert=is_dist)

    errors = {}
    for img in norm_mos.keys():
        if img in norm_iqa:
            errors[img] = abs(norm_mos[img] - norm_iqa[img])

    if not errors: continue
    best_img, worst_img = min(errors, key=errors.get), max(errors, key=errors.get)

    for match, img in [("Best", best_img), ("Worst", worst_img)]:
        print(
            f"{metric_name:<10} | {match:<5} | {img.split('.')[0]:<10} | {norm_mos[img]:.3f}    | {norm_iqa[img]:.3f}    | {errors[img]:.3f}")

# --- OBSERVER PROFILING AND GLOBAL WIN RATES ---

print("\n OBSERVER CLUSTERING AND GLOBAL PREFERENCE \n")

valid_rows = [r for r in raw_rows if r["user"] in valid_users]
df_all = pd.DataFrame(valid_rows)
df_color = df_all[df_all["method"].isin(COLOR_METHODS)].copy()

user_stats = df_color.groupby("user").agg(mean_rating=("score", "mean"), std_rating=("score", "std")).reset_index()

df_orig = df_all[df_all["method"] == "orig"][["user", "scene", "score"]].rename(columns={"score": "orig_score"})
df_color_trials = df_color.merge(df_orig, on=["user", "scene"], how="inner")
df_color_trials["delta"] = df_color_trials["score"] - df_color_trials["orig_score"]
df_color_trials["win"] = df_color_trials["score"] > df_color_trials["orig_score"]

user_pref_stats = df_color_trials.groupby("user").agg(
    win_rate_vs_orig=("win", "mean"),
    mean_delta=("delta", "mean"),
    median_delta=("delta", "median")
).reset_index()

user_stats = user_stats.merge(user_pref_stats, on="user", how="left")

cluster_cols = ["mean_rating", "std_rating"]
cluster_data = user_stats[["user"] + cluster_cols].dropna().copy()

if len(cluster_data) >= 2:
    X = StandardScaler().fit_transform(cluster_data[cluster_cols])
    cluster_data["cluster"] = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(X)

    cluster_means = cluster_data.groupby("cluster")["mean_rating"].mean().sort_values().reset_index()
    style_map = {cluster_means.loc[0, "cluster"]: "Strict", cluster_means.loc[1, "cluster"]: "Lenient"}
    cluster_data["cluster_name"] = cluster_data["cluster"].map(style_map)
    user_stats = user_stats.merge(cluster_data[["user", "cluster", "cluster_name"]], on="user", how="left")

    cluster_summary = user_stats.groupby(["cluster", "cluster_name"], as_index=False).agg(
        N=("user", "nunique"),
        Mean=("mean_rating", "mean"),
        Std=("std_rating", "mean"),
        Win_Rate=("win_rate_vs_orig", "mean"),
        Mean_Delta=("mean_delta", "mean"),
        Med_Delta=("median_delta", "mean")
    ).sort_values("cluster")

    cluster_summary["Mean"] = cluster_summary["Mean"].map("{:.3f}".format)
    cluster_summary["Std"] = cluster_summary["Std"].map("{:.3f}".format)
    cluster_summary["Win_Rate"] = (cluster_summary["Win_Rate"] * 100).map("{:.1f}%".format)
    cluster_summary["Mean_Delta"] = cluster_summary["Mean_Delta"].map("{:.3f}".format)
    cluster_summary["Med_Delta"] = cluster_summary["Med_Delta"].map("{:.3f}".format)

    print("Observer Profiles (Strict vs. Lenient):")
    print(cluster_summary.to_string(index=False))

df_pivot = df_all.pivot_table(index=["user", "scene"], columns="method", values="score", aggfunc="mean").reset_index()
if "orig" in df_pivot.columns:
    print("\nGlobal Win Rates (All Valid Observers):")
    methods_present = [m for m in ["cnn", "gan", "tf"] if m in df_pivot.columns]

    for m in methods_present:
        m_win = (df_color_trials[df_color_trials["method"] == m]["win"]).mean() * 100
        print(f"  {m.upper()}: {m_win:.1f}%")

    user_method_win = df_color_trials["win"].mean() * 100

    df_pivot["max_color"] = df_pivot[methods_present].max(axis=1)
    valid_user_scenes = df_pivot[["max_color", "orig"]].dropna()
    user_scene_win = (valid_user_scenes["max_color"] > valid_user_scenes["orig"]).mean() * 100

    print(f"\n  User-Method (Single pooled vs. ORIG): {user_method_win:.1f}%")
    print(f"  User-Scene (>=1 model beats ORIG per trial): {user_scene_win:.1f}%")

user_stats.to_csv("user_stats.csv", index=False)
if "cluster" in user_stats.columns:
    user_stats[["user", "cluster", "cluster_name"]].to_csv("user_clusters.csv", index=False)

print("\nComplete.")