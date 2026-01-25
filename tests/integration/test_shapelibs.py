from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# ----------------------------
# Fixtures
# ----------------------------

def make_fixtures(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Wide observations: dims + measures
    wide = pd.DataFrame({
        "geo_id": ["ZA-GP", "ZA-WC", "ZA-GP", "ZA-WC"],
        "sex": ["F", "F", "M", "M"],
        "age_group": ["15-24", "15-24", "15-24", "15-24"],
        "population": [1_200_000, 600_000, 1_150_000, 580_000],
        "unemployed": [180_000, 75_000, 160_000, 70_000],
        "unemployment_rate": [0.15, 0.125, 0.139, 0.121],
    })
    p1 = out_dir / "wide_observations.csv"
    wide.to_csv(p1, index=False)

    # 2) Long format: indicator + value
    long = pd.DataFrame({
        "geo_id": ["ZA-GP", "ZA-GP", "ZA-WC", "ZA-WC"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
        "indicator": ["population", "unemployed", "population", "unemployed"],
        "value": [1_200_000, 180_000, 600_000, 75_000],
    })
    p2 = out_dir / "long_indicators.csv"
    long.to_csv(p2, index=False)

    # 3) Wide-by-time columns: time encoded in columns
    wide_time = pd.DataFrame({
        "geo_id": ["ZA-GP", "ZA-WC"],
        "sex": ["F", "F"],
        "2024-01": [10.0, 7.0],
        "2024-02": [11.0, 7.5],
        "2024-03": [12.0, 8.0],
        "2024-04": [11.2, 7.8],
    })
    p3 = out_dir / "wide_time_columns.csv"
    wide_time.to_csv(p3, index=False)

    # 4) Series object column (array encoded as JSON string)
    series_col = pd.DataFrame({
        "geo_id": ["ZA-GP", "ZA-WC"],
        "series": [json.dumps([10.0, 11.0, 12.0, 11.2]), json.dumps([7.0, 7.5, 8.0, 7.8])],
        "frequency": ["monthly", "monthly"],
        "start_date": ["2024-01-01", "2024-01-01"],
    })
    p4 = out_dir / "series_column.csv"
    series_col.to_csv(p4, index=False)

    return {
        "wide_observations": p1,
        "long_indicators": p2,
        "wide_time_columns": p3,
        "series_column": p4,
    }


# ----------------------------
# Frictionless tests
# ----------------------------

def run_frictionless(csv_path: Path) -> dict[str, Any]:
    """
    Uses frictionless to infer a schema from CSV.
    """
    try:
        import frictionless as fl
    except Exception as e:
        return {"error": f"Failed to import frictionless: {e}"}

    resource = fl.Resource(path=str(csv_path))
    # infer schema
    resource.infer()

    schema = resource.schema
    fields = []
    for f in schema.fields:
        fields.append({
            "name": f.name,
            "type": f.type,               # e.g. "string", "integer", "number", "date"
            "format": getattr(f, "format", None),
            "constraints": f.constraints or {},
        })

    report = {
        "path": str(csv_path),
        "frictionless": {
            "dialect": _safe_obj(resource.dialect),
            "fields": fields,
        }
    }
    return report


# ----------------------------
# DataProfiler tests
# ----------------------------



# ----------------------------
# ydata-profiling (optional)
# ----------------------------

def run_ydata_profiling(csv_path: Path, out_html: Path) -> dict[str, Any]:
    """
    Generates an HTML profiling report. Optional.
    """
    try:
        from ydata_profiling import ProfileReport
    except Exception as e:
        return {"error": f"Failed to import ydata_profiling: {e}"}

    df = pd.read_csv(csv_path)
    report = ProfileReport(df, minimal=True, title=f"Profile: {csv_path.name}")
    report.to_file(out_html)
    return {"html_report": str(out_html)}

def _extract_dp_columns(report: Any) -> list[dict]:
    """
    Normalize DataProfiler 0.13.x 'report()' output into a list of per-column dicts.
    Handles common shapes:
      - report["data_stats"] == list[ {column profile}, ... ]
      - report["data_stats"] == [ {"column_stats": [...] } ]
      - report["data_stats"] == {"column_stats": [...] } (other versions)
    """
    if not isinstance(report, dict):
        return []

    ds = report.get("data_stats")

    # Case A: dict with column_stats
    if isinstance(ds, dict):
        col_stats = ds.get("column_stats") or ds.get("columns") or []
        return col_stats if isinstance(col_stats, list) else []

    # Case B: list
    if isinstance(ds, list):
        # Sometimes it's already a list of columns
        if ds and isinstance(ds[0], dict) and ("column_name" in ds[0] or "data_type" in ds[0]):
            return ds

        # Sometimes list contains dict with column_stats
        for item in ds:
            if isinstance(item, dict) and ("column_stats" in item or "columns" in item):
                col_stats = item.get("column_stats") or item.get("columns") or []
                return col_stats if isinstance(col_stats, list) else []
        return []

    return []

def run_dataprofiler(csv_path: Path) -> dict[str, Any]:
    try:
        from dataprofiler import Profiler
    except Exception:
        try:
            from dataprofiler.profilers.profile_builder import Profiler  # type: ignore
        except Exception as e:
            return {"error": f"Failed to import DataProfiler Profiler: {e}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read CSV with pandas: {e}"}

    try:
        profiler = Profiler(df)
        report = profiler.report()
    except Exception as e:
        return {"error": f"Failed to profile data: {e}"}

    col_stats = _extract_dp_columns(report)

    columns_out = []
    for col in col_stats:
        if not isinstance(col, dict):
            continue
        stats = col.get("statistics") or col.get("stats") or {}

        columns_out.append({
            "column_name": col.get("column_name") or col.get("name"),
            "data_type": col.get("data_type") or col.get("dtype") or col.get("type"),
            "statistics": {
                "null_count": col.get("null_count") or stats.get("null_count"),
                "unique_count": stats.get("unique_count") or col.get("unique_count"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "stddev": stats.get("stddev") or stats.get("std"),
            },
        })

    import dataprofiler as dp
    return {
        "path": str(csv_path),
        "dataprofiler": {
            "version": getattr(dp, "__version__", "unknown"),
            "global_stats": report.get("global_stats", {}) if isinstance(report, dict) else {},
            "columns": columns_out,
            "data_stats_type": type(report.get("data_stats")).__name__ if isinstance(report, dict) else type(report).__name__,
        },
    }




# ----------------------------
# Main runner
# ----------------------------

def main():
    try:
        import dataprofiler as dp
        print("DataProfiler version:", getattr(dp, "__version__", "unknown"))
    except Exception:
        pass

    out_dir = Path("tmp_shape_lib_tests")
    fixtures = make_fixtures(out_dir)

    results: dict[str, Any] = {"fixtures": {k: str(v) for k, v in fixtures.items()}, "runs": {}}

    for name, path in fixtures.items():
        print(f"\n=== Fixture: {name} ({path.name}) ===")

        fr = run_frictionless(path)
        dp = run_dataprofiler(path)

        results["runs"][name] = {
            "frictionless": fr.get("frictionless") or fr,
            "dataprofiler": dp.get("dataprofiler") or dp,
        }

        # Print a compact summary
        if "error" in fr:
            print("Frictionless error:", fr["error"])
        else:
            print("Frictionless field types:")
            for f in fr["frictionless"]["fields"]:
                print(f"  - {f['name']}: {f['type']}")

        if "error" in dp:
            print("DataProfiler error:", dp["error"])
        else:
            print("DataProfiler column types:")
            for c in dp["dataprofiler"]["columns"]:
                print(f"  - {c['column_name']}: {c['data_type']}")

    # Optional ydata-profiling reports
    try:
        import ydata_profiling  # noqa
        do_ydata = True
    except Exception:
        do_ydata = False

    if do_ydata:
        results["ydata_reports"] = {}
        for name, path in fixtures.items():
            html_out = out_dir / f"{name}_profile.html"
            r = run_ydata_profiling(path, html_out)
            results["ydata_reports"][name] = r
            if "error" in r:
                print(f"ydata-profiling error for {name}:", r["error"])
            else:
                print(f"ydata-profiling wrote: {r['html_report']}")

    # Save JSON result bundle for inspection
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote full results bundle to: {json_path.resolve()}")

def _safe_obj(obj):
    """Best-effort JSON-serializable representation across frictionless versions."""
    if obj is None:
        return None
    if hasattr(obj, "to_descriptor"):
        try:
            return obj.to_descriptor()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out or str(obj)


if __name__ == "__main__":
    main()


