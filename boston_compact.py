from collections import Counter
from functools import reduce
from pathlib import Path

import pandas as pd


COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]
KEY_COLUMNS = ["CRIM", "RM", "AGE", "TAX", "PTRATIO", "LSTAT", "MEDV"]
ZERO_AS_MISSING = {"RM", "AGE", "TAX", "PTRATIO", "LSTAT", "MEDV"}
STAT_COLUMNS = ["CRIM", "RM", "AGE", "MEDV", "PricePerRoom"]


def read_dataset(path):
    try:
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        if list(df.columns) == COLUMNS:
            pass
        elif df.shape[1] == 14:
            df.columns = COLUMNS
        elif df.shape[1] == 1:
            df = df.iloc[:, 0].astype(str).str.split(expand=True)
            if df.shape[1] != 14:
                raise ValueError("Dataset could not be converted to 14 columns.")
            df.columns = COLUMNS
        else:
            df = pd.DataFrame([list(df.columns), *df.values.tolist()])
            if df.shape[1] != 14:
                raise ValueError("Dataset format not recognized.")
            df.columns = COLUMNS
        for col in COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Dataset loaded successfully from: {path}")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
    except PermissionError:
        print(f"Permission denied while reading: {path}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        print(f"Dataset format error: {error}")
    except OSError as error:
        print(f"Operating system error: {error}")
    except Exception as error:
        print(f"Unexpected error: {error}")
    return None


def write_text(path, content):
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return True
    except Exception as error:
        print(f"Could not write file {path}: {error}")
        return False


def mean_loop(values):
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0


def recursive_sum(values):
    if not values:
        return 0.0
    return float(values[0]) + recursive_sum(values[1:])


def median_loop(values):
    values = sorted(values)
    mid = len(values) // 2
    return (values[mid - 1] + values[mid]) / 2 if len(values) % 2 == 0 else values[mid]


def mode_loop(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    top = max(counts.values())
    return [value for value, count in counts.items() if count == top]


def min_loop(values):
    smallest = values[0]
    for value in values[1:]:
        if value < smallest:
            smallest = value
    return smallest


def max_loop(values):
    largest = values[0]
    for value in values[1:]:
        if value > largest:
            largest = value
    return largest


def age_bucket(age):
    if age < 30:
        return "New"
    if age <= 70:
        return "Old"
    return "VeryOld"


def neighborhood_bucket(rad, tax):
    if rad >= 10 and tax >= 400:
        return "HighAccessHighTax"
    if rad >= 10:
        return "HighAccessLowTax"
    if tax >= 400:
        return "LowAccessHighTax"
    return "LowAccessLowTax"


def explore(df):
    print("\n" + "=" * 70)
    print("FIRST 10 ROWS")
    print("=" * 70)
    print(df.head(10))
    print("\n" + "=" * 70)
    print("LAST 5 ROWS")
    print("=" * 70)
    print(df.tail(5))
    print("\n" + "=" * 70)
    print("DATAFRAME INFO")
    print("=" * 70)
    df.info()
    print("\n" + "=" * 70)
    print("HOUSES WHERE RM > 7 OR MEDV > 25")
    print("=" * 70)
    for i in range(len(df)):
        if df.loc[i, "RM"] > 7 or df.loc[i, "MEDV"] > 25:
            print(
                f"Row {i}: RM={df.loc[i, 'RM']}, MEDV={df.loc[i, 'MEDV']}, "
                f"CRIM={df.loc[i, 'CRIM']}, TAX={df.loc[i, 'TAX']}"
            )


def clean_and_derive(df):
    summary = {}
    print("\n" + "=" * 70)
    print("MISSING VALUE HANDLING")
    print("=" * 70)
    for col in KEY_COLUMNS:
        valid = []
        missing = 0
        zeroes = 0
        for value in df[col]:
            if pd.isna(value):
                missing += 1
                continue
            if value == 0:
                zeroes += 1
                if col in ZERO_AS_MISSING:
                    continue
            valid.append(float(value))
        avg = mean_loop(valid)
        updated = []
        replaced = 0
        for value in df[col]:
            if pd.isna(value) or (value == 0 and col in ZERO_AS_MISSING):
                updated.append(avg)
                replaced += 1
            else:
                updated.append(value)
        df[col] = updated
        summary[col] = {
            "missing_count": missing,
            "zero_count": zeroes,
            "manual_mean_used": avg,
            "replaced_count": replaced,
        }
        print(f"{col}: missing={missing}, zero={zeroes}, mean={avg:.4f}, replaced={replaced}")
    price_per_room = []
    tax_adjusted = []
    house_age = []
    for i in range(len(df)):
        medv = float(df.loc[i, "MEDV"])
        rm = float(df.loc[i, "RM"])
        tax = float(df.loc[i, "TAX"])
        age = float(df.loc[i, "AGE"])
        price_per_room.append(0.0 if rm == 0 else medv / rm)
        tax_adjusted.append(medv - (tax * 0.01 * medv))
        house_age.append(age_bucket(age))
    df["PricePerRoom"] = price_per_room
    df["TaxAdjustedPrice"] = tax_adjusted
    df["HouseAgeCategory"] = house_age
    print("\nCreated derived columns: PricePerRoom, TaxAdjustedPrice, HouseAgeCategory")
    return df, summary


def get_price_range():
    try:
        low = float(input("\nEnter minimum MEDV price: ").strip())
        high = float(input("Enter maximum MEDV price: ").strip())
    except (ValueError, EOFError):
        print("Invalid input. Using default range 20 to 30.")
        low, high = 20.0, 30.0
    except KeyboardInterrupt:
        print("\nInput cancelled. Using default range 20 to 30.")
        low, high = 20.0, 30.0
    if low > high:
        low, high = high, low
    return low, high


def search_by_price(df, low, high, limit=10):
    print("\n" + "=" * 70)
    print(f"HOUSES WITH MEDV BETWEEN {low} AND {high}")
    print("=" * 70)
    matches = []
    i = 0
    while i < len(df):
        medv = float(df.loc[i, "MEDV"])
        if medv < low:
            i += 1
            continue
        if medv > high:
            i += 1
            continue
        row = {
            "row_index": i,
            "MEDV": medv,
            "RM": float(df.loc[i, "RM"]),
            "CRIM": float(df.loc[i, "CRIM"]),
            "HouseAgeCategory": df.loc[i, "HouseAgeCategory"],
        }
        matches.append(row)
        print(f"Match {len(matches)}: {row}")
        if len(matches) >= limit:
            print("Sufficient results found. Stopping search with break.")
            break
        i += 1
    if not matches:
        print("No houses were found in the selected range.")
    return matches


def neighborhood_stats(df):
    groups = {}
    for i in range(len(df)):
        key = neighborhood_bucket(float(df.loc[i, "RAD"]), float(df.loc[i, "TAX"]))
        groups.setdefault(key, {"MEDV": 0.0, "CRIM": 0.0, "RM": 0.0, "count": 0})
        groups[key]["MEDV"] += float(df.loc[i, "MEDV"])
        groups[key]["CRIM"] += float(df.loc[i, "CRIM"])
        groups[key]["RM"] += float(df.loc[i, "RM"])
        groups[key]["count"] += 1
    result = {}
    for key, item in groups.items():
        result[key] = {
            "average_MEDV": item["MEDV"] / item["count"],
            "average_CRIM": item["CRIM"] / item["count"],
            "average_RM": item["RM"] / item["count"],
        }
    print("\nNeighborhood statistics dictionary created successfully.")
    return result


def weighted_averages(df):
    medv = [float(x) for x in df["MEDV"].tolist()]
    rm_weights = [float(x) for x in df["RM"].tolist()]
    crim_weights = [1 / (float(x) + 1) for x in df["CRIM"].tolist()]

    def with_reduce(values, weights):
        pairs = list(zip(values, weights))
        num = reduce(lambda acc, item: acc + item[0] * item[1], pairs, 0.0)
        den = reduce(lambda acc, item: acc + item[1], pairs, 0.0)
        return num / den if den else 0.0

    def with_loop(values, weights):
        num = 0.0
        den = 0.0
        for i in range(len(values)):
            num += values[i] * weights[i]
            den += weights[i]
        return num / den if den else 0.0

    return {
        "reduce_weighted_by_rm": with_reduce(medv, rm_weights),
        "manual_weighted_by_rm": with_loop(medv, rm_weights),
        "reduce_weighted_by_inverse_crim": with_reduce(medv, crim_weights),
        "manual_weighted_by_inverse_crim": with_loop(medv, crim_weights),
    }


def stats_summary(df):
    result = {}
    for col in STAT_COLUMNS:
        values = [float(x) for x in df[col].tolist()]
        result[col] = {
            "manual": {
                "mean": recursive_sum(values) / len(values),
                "median": median_loop(values),
                "mode": mode_loop(values),
                "min": min_loop(values),
                "max": max_loop(values),
            },
            "pandas": {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "mode": [float(x) for x in df[col].mode().tolist()],
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            },
            "unique_count": len(set(tuple(values))),
            "frequency_top_3": Counter(values).most_common(3),
        }
    return result


def grouped_stats(df):
    result = {"HouseAgeCategory": {}, "CHAS": {}}
    for category in set(df["HouseAgeCategory"]):
        part = df[df["HouseAgeCategory"] == category]
        result["HouseAgeCategory"][category] = {
            "average_MEDV": float(part["MEDV"].mean()),
            "average_CRIM": float(part["CRIM"].mean()),
            "average_RM": float(part["RM"].mean()),
            "count": int(len(part)),
        }
    for chas in set(df["CHAS"]):
        part = df[df["CHAS"] == chas]
        result["CHAS"][chas] = {
            "average_MEDV": float(part["MEDV"].mean()),
            "average_CRIM": float(part["CRIM"].mean()),
            "average_RM": float(part["RM"].mean()),
            "count": int(len(part)),
        }
    return result


def build_report(df, cleaning, matches, neighborhoods, weighted, stats, grouped, low, high):
    lines = ["BOSTON HOUSING COMPACT REPORT", "=" * 80, f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns", ""]
    lines += ["1. First 10 Rows", df.head(10).to_string(), ""]
    lines += ["2. Last 5 Rows", df.tail(5).to_string(), ""]
    lines += ["3. Missing Value Handling"]
    lines += [f"{col}: {info}" for col, info in cleaning.items()]
    lines += ["", "4. Derived Columns", df[["MEDV", "RM", "PricePerRoom", "TaxAdjustedPrice", "HouseAgeCategory"]].head(10).to_string(), ""]
    lines += [f"5. Price Range Search ({low} to {high})"]
    lines += [str(x) for x in matches] if matches else ["No houses found."]
    lines += ["", "6. Neighborhood Statistics"]
    lines += [f"{k}: {v}" for k, v in neighborhoods.items()]
    lines += ["", "7. Weighted Averages"]
    lines += [f"{k}: {v:.4f}" for k, v in weighted.items()]
    lines += ["", "8. Manual vs Pandas Statistics"]
    for col, info in stats.items():
        lines += [f"{col}: {info}", ""]
    lines += ["9. Grouped Statistics by HouseAgeCategory"]
    lines += [f"{k}: {v}" for k, v in grouped["HouseAgeCategory"].items()]
    lines += ["", "10. Grouped Statistics by CHAS"]
    lines += [f"{k}: {v}" for k, v in grouped["CHAS"].items()]
    return "\n".join(lines)


def main():
    folder = Path(__file__).resolve().parent
    df = read_dataset(folder / "Boston.csv")
    if df is None:
        return
    explore(df)
    df, cleaning = clean_and_derive(df)
    low, high = get_price_range()
    matches = search_by_price(df, low, high)
    neighborhoods = neighborhood_stats(df)
    weighted = weighted_averages(df)
    stats = stats_summary(df)
    grouped = grouped_stats(df)
    report = build_report(df, cleaning, matches, neighborhoods, weighted, stats, grouped, low, high)
    if write_text(folder / "boston_compact_report.txt", report):
        print("\nCompact report saved successfully to: boston_compact_report.txt")


if __name__ == "__main__":
    main()
