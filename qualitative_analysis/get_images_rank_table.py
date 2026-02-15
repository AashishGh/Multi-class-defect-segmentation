# import os
# import pandas as pd
# from typing import Dict, List, Optional

# def create_iou_it_table_csv(
#     model_csv_map: Dict[str, str],
#     out_csv_path: str,
#     image_ids: Optional[List[str]] = None,
#     filename_col: str = "filename",
#     value_col: str = "iou_it",
# ) -> pd.DataFrame:
#     """
#     Create a CSV with columns:
#       image_id, iou_it_<model1>, iou_it_<model2>, ...

#     Args:
#         model_csv_map: dict like {
#             "doubleunet": "results/doubleunet/...csv",
#             "transrupnet": "results/transrupnet/...csv",
#             "transnetr": "results/transnetr/...csv",
#             "vmunetv2": "results/vmunetv2/...csv"
#         }
#         out_csv_path: path to save new table
#         image_ids: list of image_ids to include. If None, uses union of all ids across CSVs.
#         filename_col: column containing image ids (default "filename")
#         value_col: metric column to extract (default "iou_it")

#     Returns:
#         df_out: merged dataframe
#     """
#     # Validate CSV existence and required columns
#     per_model_value_map: Dict[str, Dict[str, float]] = {}
#     all_ids = set()

#     for model_name, csv_path in model_csv_map.items():
#         if not os.path.exists(csv_path):
#             raise FileNotFoundError(f"CSV not found for model '{model_name}': {csv_path}")

#         df = pd.read_csv(csv_path)

#         if filename_col not in df.columns:
#             raise KeyError(
#                 f"Column '{filename_col}' not found in '{csv_path}'. Columns: {list(df.columns)}"
#             )
#         if value_col not in df.columns:
#             raise KeyError(
#                 f"Column '{value_col}' not found in '{csv_path}'. Columns: {list(df.columns)}"
#             )

#         # build image_id -> iou_it lookup (first occurrence wins)
#         ids = df[filename_col].astype(str).tolist()
#         vals = pd.to_numeric(df[value_col], errors="coerce").tolist()

#         lookup: Dict[str, float] = {}
#         for img_id, v in zip(ids, vals):
#             if img_id not in lookup:
#                 lookup[img_id] = v

#         per_model_value_map[model_name] = lookup
#         all_ids.update(ids)

#     # Determine image list
#     if image_ids is None:
#         image_ids = sorted(all_ids)
#     else:
#         image_ids = [str(x) for x in image_ids]

#     # Build output rows
#     model_order = list(model_csv_map.keys())  # preserves insertion order
#     rows = []

#     for img_id in image_ids:
#         row = {"image_id": img_id}
#         for m in model_order:
#             row[f"{value_col}_{m}"] = per_model_value_map[m].get(img_id, pd.NA)
#         rows.append(row)

#     df_out = pd.DataFrame(rows)

#     # Save
#     os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
#     df_out.to_csv(out_csv_path, index=False)
#     return df_out

# MODEL_CSV_MAP = {
#     "doubleunet": "results/doubleunet/qualitative/per_image_iou_test_dataset.csv",
#     "transrupnet": "results/transrupnet/qualitative/per_image_iou_test_dataset.csv",
#     "transnetr": "results/transnetr/qualitative/per_image_iou_test_dataset.csv",
#     "vmunetv2": "results/vmunetv2/qualitative/per_image_iou_test_dataset.csv",
# }

# df = create_iou_it_table_csv(
#     model_csv_map=MODEL_CSV_MAP,
#     out_csv_path="results/qualitative/iou_caries_table_4models.csv",
#     image_ids=None,          # union of all images
#     filename_col="filename",
#     value_col="iou_caries",
# )
# print(df.head())

import pandas as pd
from typing import Dict

def add_impact_score_and_sort(
    in_csv_path: str,
    out_csv_path: str = None,
    weights: Dict[str, float] = None,
) -> str:
    """
    Adds `impact_score` = weighted average of iou_it across 4 model columns and
    sorts CSV by `impact_score` descending.

    Expected columns in input CSV:
      - image_id
      - iou_it_doubleunet
      - iou_it_transrupnet
      - iou_it_transnetr
      - iou_it_vmunetv2

    Default weights (doubleunet -> vmunetv2): 0.3, 0.28, 0.22, 0.2

    Args:
        in_csv_path: existing merged iou_it table
        out_csv_path: where to save (if None, overwrite in_csv_path)
        weights: optional dict with keys:
            {"doubleunet":0.3, "transrupnet":0.28, "transnetr":0.22, "vmunetv2":0.2}

    Returns:
        Path to the saved CSV.
    """
    if weights is None:
        weights = {"doubleunet": 0.30, "transrupnet": 0.28, "transnetr": 0.22, "vmunetv2": 0.20}

    df = pd.read_csv(in_csv_path)

    # Map model -> column name in CSV
    col_map = {
        "doubleunet": "iou_caries_doubleunet",
        "transrupnet": "iou_caries_transrupnet",
        "transnetr": "iou_caries_transnetr",
        "vmunetv2": "iou_caries_vmunetv2",
    }

    # Validate & numeric cast
    for m, col in col_map.items():
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {in_csv_path}. Found: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Weighted average (normalize by sum of weights for non-missing entries per row)
    w_sum = None
    w_num = None
    for m, w in weights.items():
        col = col_map[m]
        present = df[col].notna().astype(float)  # 1 if value exists else 0
        if w_sum is None:
            w_sum = w * present
            w_num = w * df[col].fillna(0.0)
        else:
            w_sum = w_sum + (w * present)
            w_num = w_num + (w * df[col].fillna(0.0))

    df["impact_score"] = (w_num / w_sum).where(w_sum > 0, pd.NA)

    # Sort by new field (desc)
    df = df.sort_values("impact_score", ascending=False, na_position="last").reset_index(drop=True)

    save_path = out_csv_path or in_csv_path
    df.to_csv(save_path, index=False)
    return save_path

add_impact_score_and_sort(
    in_csv_path="results/qualitative/iou_caries_table_4models.csv",
    out_csv_path="results/qualitative/iou_caries_table_4models_with_impact.csv"
)
