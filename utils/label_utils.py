
import pandas as pd


def get_label_id(label_value, label_map):
    if isinstance(label_value, int):
        return label_value

    label_value = str(label_value).strip().lower()

    if label_value not in label_map:
        raise ValueError(
            f"Unknown label: {label_value}. "
            f"Known labels: {list(label_map.keys())}"
        )

    return label_map[label_value]



def oversample_minority_classes(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    if len(df) == 0:
        return df

    counts = df[label_col].value_counts()
    max_count = counts.max()

    parts = []

    for label, group in df.groupby(label_col):
        if len(group) < max_count:
            extra = group.sample(
                n=max_count - len(group),
                replace=True,
                random_state=random_state,
            )
            group = pd.concat([group, extra], ignore_index=True)

        parts.append(group)

    balanced_df = pd.concat(parts, ignore_index=True)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return balanced_df
