#!/usr/bin/env python3
"""Build ECCV-friendly PDF tables from compare_real_images_detectors CSV output.

Features:
- Parses real-world filename factors (distance / day-night / UV-baseline / replicate)
- Treats UV-only filenames (e.g., 02m_UV_1.png) as night UV with headlights off
- Aggregates duplicate captures (_1/_2/_3) as mean +/- std
- Exports paper-ready PDF tables (and CSVs for reproducibility)

Outputs (in --out-dir):
- overall_summary.csv / overall_summary.pdf
- by_condition_long.csv / by_condition_long.pdf
- by_condition_pivot_target_conf.csv / by_condition_pivot_target_conf.pdf
- by_condition_pivot_detect_rate.csv / by_condition_pivot_detect_rate.pdf
- paired_uv_vs_baseline_long.csv / paired_uv_vs_baseline_long.pdf
- paired_uv_vs_baseline_pivot_delta_target_conf.csv / paired_uv_vs_baseline_pivot_delta_target_conf.pdf
- parse_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for PDF export. Install it in your environment. "
        f"Import error: {e}"
    )


# ------------------------- parsing / normalization -------------------------

def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float('nan')


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'true', '1', 'yes', 'y'}:
        return True
    if s in {'false', '0', 'no', 'n', ''}:
        return False
    return bool(v)


def _mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    return (sum(arr) / len(arr)) if arr else float('nan')


def _std(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if not arr:
        return float('nan')
    mu = sum(arr) / len(arr)
    return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))


def _median(vals: Iterable[float]) -> float:
    arr = sorted(float(v) for v in vals if not math.isnan(float(v)))
    n = len(arr)
    if n == 0:
        return float('nan')
    m = n // 2
    return arr[m] if n % 2 == 1 else 0.5 * (arr[m - 1] + arr[m])


def _fmt_num(v: float, prec: int = 3) -> str:
    return '' if math.isnan(v) else f"{v:.{prec}f}"


def _fmt_pm(mu: float, sd: float, prec: int = 3) -> str:
    if math.isnan(mu):
        return ''
    if math.isnan(sd):
        return f"{mu:.{prec}f}"
    return f"{mu:.{prec}f} ± {sd:.{prec}f}"


def _fmt_pct_pm(mu: float, sd: float, prec: int = 1) -> str:
    if math.isnan(mu):
        return ''
    if math.isnan(sd):
        return f"{100*mu:.{prec}f}%"
    return f"{100*mu:.{prec}f}% ± {100*sd:.{prec}f}%"


def _parse_filename(image_path: str) -> dict[str, Any]:
    name = str(image_path).replace('\\', '/').split('/')[-1]
    stem = name.rsplit('.', 1)[0]
    tokens = [t for t in re.split(r'[_\s]+', stem) if t]

    out = {
        'file_stem': stem,
        'distance_m': None,
        'ambient': 'unknown',    # day/night
        'uv_state': 'unknown',   # baseline/uv
        'headlights': 'unknown', # on/off/na
        'rep': None,
        'condition': 'unknown',
        'parse_ok': False,
        'parse_note': '',
    }

    # distance token (02m / 4m)
    if tokens:
        m = re.match(r'^0*(\d+)m$', tokens[0].lower())
        if m:
            out['distance_m'] = int(m.group(1))
            tokens = tokens[1:]
    if out['distance_m'] is None:
        m = re.search(r'(?<!\d)(\d+)m', stem.lower())
        if m:
            out['distance_m'] = int(m.group(1))

    rep = None
    norm_tokens: list[str] = []
    for t in tokens:
        tl = t.lower()
        # handles UV1 / Baseline2 / Day3
        m = re.match(r'^(uv|baseline|base|day|night|daytime|nighttime)(\d+)$', tl)
        if m:
            norm_tokens.append(m.group(1))
            rep = int(m.group(2))
            continue
        if re.fullmatch(r'\d+', tl):
            rep = int(tl)
            continue
        norm_tokens.append(tl)
    out['rep'] = rep

    for t in norm_tokens:
        if t in {'day', 'daytime'}:
            out['ambient'] = 'day'
        elif t in {'night', 'nighttime'}:
            out['ambient'] = 'night'

    for t in norm_tokens:
        if t in {'baseline', 'base'}:
            out['uv_state'] = 'baseline'
        elif t == 'uv':
            out['uv_state'] = 'uv'

    # user-specific interpretation: UV-only means UV at night with headlights off
    if out['ambient'] == 'unknown' and out['uv_state'] == 'uv':
        out['ambient'] = 'night'
        out['headlights'] = 'off'
        out['parse_note'] = 'uv_only_interpreted_as_night_uv_no_headlights'
    elif out['ambient'] == 'day':
        out['headlights'] = 'na'
    elif out['ambient'] == 'night':
        out['headlights'] = 'on'

    if out['ambient'] == 'day' and out['uv_state'] == 'baseline':
        out['condition'] = 'day_baseline'
    elif out['ambient'] == 'day' and out['uv_state'] == 'uv':
        out['condition'] = 'day_uv'
    elif out['ambient'] == 'night' and out['uv_state'] == 'baseline' and out['headlights'] == 'on':
        out['condition'] = 'night_baseline_headlights'
    elif out['ambient'] == 'night' and out['uv_state'] == 'uv' and out['headlights'] == 'on':
        out['condition'] = 'night_uv_headlights'
    elif out['ambient'] == 'night' and out['uv_state'] == 'uv' and out['headlights'] == 'off':
        out['condition'] = 'night_uv_no_headlights'
    else:
        out['condition'] = f"{out['ambient']}_{out['uv_state']}_{out['headlights']}"

    out['parse_ok'] = (out['distance_m'] is not None and out['uv_state'] in {'baseline', 'uv'})
    if not out['parse_ok'] and not out['parse_note']:
        out['parse_note'] = 'incomplete_filename_pattern'
    return out


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            row = dict(rec)
            row.update(_parse_filename(rec.get('image_path', '')))
            row['target_conf'] = _safe_float(row.get('target_conf'))
            row['top_conf'] = _safe_float(row.get('top_conf'))
            row['runtime_ms'] = _safe_float(row.get('runtime_ms'))
            row['target_missing'] = _safe_bool(row.get('target_missing'))
            row['top_misclass'] = _safe_bool(row.get('top_misclass'))
            row['top_is_target'] = _safe_bool(row.get('top_is_target'))
            rows.append(row)
    return rows


# ------------------------------- aggregation -------------------------------

def _agg_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    target = [float(r['target_conf']) for r in rows]
    top = [float(r['top_conf']) for r in rows]
    runtime = [float(r['runtime_ms']) for r in rows]
    miss = [1.0 if bool(r['target_missing']) else 0.0 for r in rows]
    detect = [1.0 - m for m in miss]
    miscls = [1.0 if bool(r['top_misclass']) else 0.0 for r in rows]
    return {
        'n': len(rows),
        'mean_target_conf': _mean(target),
        'std_target_conf': _std(target),
        'median_target_conf': _median(target),
        'mean_top_conf': _mean(top),
        'std_top_conf': _std(top),
        'target_detect_rate': _mean(detect),
        'std_target_detect_rate': _std(detect),
        'target_missing_rate': _mean(miss),
        'std_target_missing_rate': _std(miss),
        'top_misclass_rate': _mean(miscls),
        'std_top_misclass_rate': _std(miscls),
        'mean_runtime_ms': _mean(runtime),
        'std_runtime_ms': _std(runtime),
    }


def _group(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def _overall_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _group(rows, ['detector_name'])
    out = []
    for (det,), g in sorted(groups.items(), key=lambda kv: str(kv[0][0])):
        s = _agg_stats(g)
        out.append({
            'Detector': det,
            'N': int(s['n']),
            'Target Conf': _fmt_pm(s['mean_target_conf'], s['std_target_conf'], 3),
            'Detect Rate': _fmt_pct_pm(s['target_detect_rate'], s['std_target_detect_rate'], 1),
            'Miss Rate': _fmt_pct_pm(s['target_missing_rate'], s['std_target_missing_rate'], 1),
            'Top Misclass': _fmt_pct_pm(s['top_misclass_rate'], s['std_top_misclass_rate'], 1),
            'Top Conf': _fmt_pm(s['mean_top_conf'], s['std_top_conf'], 3),
            'Runtime (ms)': _fmt_pm(s['mean_runtime_ms'], s['std_runtime_ms'], 1),
        })
    return out


def _by_condition_long_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _group(rows, ['detector_name', 'distance_m', 'condition', 'ambient', 'headlights', 'uv_state'])
    out = []
    for key, g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), 999 if kv[0][1] is None else kv[0][1], str(kv[0][2]))):
        det, dist, cond, ambient, hl, uv = key
        s = _agg_stats(g)
        out.append({
            'Detector': det,
            'Dist (m)': dist,
            'Condition': cond,
            'Ambient': ambient,
            'HL': hl,
            'Mode': uv,
            'N': int(s['n']),
            'Target Conf': _fmt_pm(s['mean_target_conf'], s['std_target_conf'], 3),
            'Detect Rate': _fmt_pct_pm(s['target_detect_rate'], s['std_target_detect_rate'], 1),
            'Top Misclass': _fmt_pct_pm(s['top_misclass_rate'], s['std_top_misclass_rate'], 1),
            'Runtime (ms)': _fmt_pm(s['mean_runtime_ms'], s['std_runtime_ms'], 1),
        })
    return out


def _paired_uv_vs_baseline(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # pair baseline and uv rows by detector+distance+ambient+headlights+rep
    base = {}
    uv = {}
    for r in rows:
        if not bool(r.get('parse_ok', False)):
            continue
        rep = r.get('rep')
        if rep is None:
            continue
        k = (r.get('detector_name'), r.get('distance_m'), r.get('ambient'), r.get('headlights'), rep)
        if r.get('uv_state') == 'baseline':
            base[k] = r
        elif r.get('uv_state') == 'uv':
            uv[k] = r

    paired_rows: list[dict[str, Any]] = []
    for k in sorted(set(base.keys()) & set(uv.keys()), key=lambda x: (str(x[0]), 999 if x[1] is None else x[1], str(x[2]), str(x[3]), int(x[4]))):
        b = base[k]
        u = uv[k]
        det, dist, ambient, hl, rep = k
        paired_rows.append({
            'Detector': det,
            'Dist (m)': dist,
            'Ambient': ambient,
            'HL': hl,
            'Rep': rep,
            'Baseline Target Conf': float(b['target_conf']),
            'UV Target Conf': float(u['target_conf']),
            'Δ Target Conf (UV-B)': float(u['target_conf']) - float(b['target_conf']),
            'Baseline Miss': 1.0 if bool(b['target_missing']) else 0.0,
            'UV Miss': 1.0 if bool(u['target_missing']) else 0.0,
            'Δ Miss (UV-B)': (1.0 if bool(u['target_missing']) else 0.0) - (1.0 if bool(b['target_missing']) else 0.0),
            'Baseline Misclass': 1.0 if bool(b['top_misclass']) else 0.0,
            'UV Misclass': 1.0 if bool(u['top_misclass']) else 0.0,
            'Δ Misclass (UV-B)': (1.0 if bool(u['top_misclass']) else 0.0) - (1.0 if bool(b['top_misclass']) else 0.0),
        })

    # compact aggregate across duplicate reps
    agg_groups = _group(paired_rows, ['Detector', 'Dist (m)', 'Ambient', 'HL'])
    agg_rows: list[dict[str, Any]] = []
    for key, g in sorted(agg_groups.items(), key=lambda kv: (str(kv[0][0]), 999 if kv[0][1] is None else kv[0][1], str(kv[0][2]), str(kv[0][3]))):
        det, dist, ambient, hl = key
        dtc = [float(r['Δ Target Conf (UV-B)']) for r in g]
        dmiss = [float(r['Δ Miss (UV-B)']) for r in g]
        dmis = [float(r['Δ Misclass (UV-B)']) for r in g]
        agg_rows.append({
            'Detector': det,
            'Dist (m)': dist,
            'Ambient': ambient,
            'HL': hl,
            'N pairs': len(g),
            'Δ Target Conf (UV-B)': _fmt_pm(_mean(dtc), _std(dtc), 3),
            'Δ Miss Rate (UV-B)': _fmt_pct_pm(_mean(dmiss), _std(dmiss), 1),
            'Δ Top Misclass (UV-B)': _fmt_pct_pm(_mean(dmis), _std(dmis), 1),
        })
    return paired_rows, agg_rows


def _pivot(rows: list[dict[str, Any]], row_key: str, col_key: str, value_key: str) -> list[dict[str, Any]]:
    row_vals = sorted({str(r.get(row_key, '')) for r in rows})
    col_vals = sorted({str(r.get(col_key, '')) for r in rows})
    lookup = {(str(r.get(row_key, '')), str(r.get(col_key, ''))): r.get(value_key, '') for r in rows}
    out = []
    for rv in row_vals:
        rec = {row_key: rv}
        for cv in col_vals:
            rec[cv] = lookup.get((rv, cv), '')
        out.append(rec)
    return out


def _by_condition_pivots(by_cond_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    for r in by_cond_rows:
        dist = r.get('Dist (m)')
        cond = r.get('Condition')
        col = f"{int(dist):02d}m | {cond}" if dist not in (None, '') else str(cond)
        normalized.append({
            'Detector': r['Detector'],
            'Col': col,
            'Target Conf': r['Target Conf'],
            'Detect Rate': r['Detect Rate'],
        })
    return _pivot(normalized, 'Detector', 'Col', 'Target Conf'), _pivot(normalized, 'Detector', 'Col', 'Detect Rate')


def _paired_pivot_delta_target(paired_agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for r in paired_agg_rows:
        dist = r.get('Dist (m)')
        ambient = r.get('Ambient')
        hl = r.get('HL')
        col = f"{int(dist):02d}m | {ambient} | HL:{hl}" if dist not in (None, '') else f"{ambient} | HL:{hl}"
        normalized.append({
            'Detector': r['Detector'],
            'Col': col,
            'Δ Target Conf (UV-B)': r['Δ Target Conf (UV-B)'],
        })
    return _pivot(normalized, 'Detector', 'Col', 'Δ Target Conf (UV-B)')


# ---------------------------- output / rendering ---------------------------

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(rows[0].keys())
    extras = sorted({k for r in rows for k in r.keys()} - set(fields))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields + extras)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pdf_table(rows: list[dict[str, Any]], out_pdf: Path, title: str, subtitle: str = '', rows_per_page: int = 20) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with PdfPages(out_pdf) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ax.text(0.04, 0.92, title, fontsize=16, fontweight='bold', ha='left', va='top')
            if subtitle:
                ax.text(0.04, 0.88, subtitle, fontsize=10, color='#4b5563', ha='left', va='top')
            ax.text(0.04, 0.80, 'No rows.', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        return

    cols = list(rows[0].keys())
    data = [[str(r.get(c, '')) for c in cols] for r in rows]

    # split across pages by row count
    chunks = [data[i:i + rows_per_page] for i in range(0, len(data), rows_per_page)]
    with PdfPages(out_pdf) as pdf:
        for page_idx, chunk in enumerate(chunks, start=1):
            fig_h = 2.0 + 0.34 * (len(chunk) + 1)
            fig_h = min(max(fig_h, 6.5), 13.0)
            fig = plt.figure(figsize=(14, fig_h))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.axis('off')

            ax.text(0.0, 1.03, title, fontsize=15, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)
            sub = subtitle
            if len(chunks) > 1:
                sub = (subtitle + ' | ' if subtitle else '') + f'Page {page_idx}/{len(chunks)}'
            if sub:
                ax.text(0.0, 0.995, sub, fontsize=9.5, color='#4b5563', ha='left', va='top', transform=ax.transAxes)

            # width heuristic based on max text lengths
            max_lens = []
            for j, c in enumerate(cols):
                m = max([len(c)] + [len(row[j]) for row in chunk])
                max_lens.append(max(6, min(m, 36)))
            total = sum(max_lens)
            col_widths = [w / total for w in max_lens]

            tbl = ax.table(
                cellText=chunk,
                colLabels=cols,
                colWidths=col_widths,
                loc='upper left',
                cellLoc='left',
                colLoc='left',
                bbox=[0.0, 0.0, 1.0, 0.94],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.8)
            tbl.scale(1, 1.2)

            # style header + zebra rows
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor('#d1d5db')
                cell.set_linewidth(0.6)
                if r == 0:
                    cell.set_facecolor('#e5e7eb')
                    cell.set_text_props(weight='bold', color='#111827')
                    cell.set_height(cell.get_height() * 1.15)
                else:
                    cell.set_facecolor('#ffffff' if (r % 2 == 1) else '#f9fafb')
                    # right align compact numeric-like columns if short header contains these keywords
                    header = cols[c].lower()
                    if any(k in header for k in ['conf', 'rate', 'runtime', 'n', 'dist', 'Δ', 'delta']):
                        cell._loc = 'right'

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def _write_parse_summary(path: Path, rows: list[dict[str, Any]], outputs: dict[str, str]) -> None:
    notes = defaultdict(int)
    conds = defaultdict(int)
    ok = 0
    for r in rows:
        if bool(r.get('parse_ok', False)):
            ok += 1
        notes[str(r.get('parse_note', ''))] += 1
        conds[str(r.get('condition', 'unknown'))] += 1
    payload = {
        'n_rows': len(rows),
        'n_parse_ok': ok,
        'n_parse_fail': len(rows) - ok,
        'conditions': dict(sorted(conds.items())),
        'parse_notes': dict(sorted(notes.items())),
        'outputs': outputs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> int:
    p = argparse.ArgumentParser(description='Create ECCV-style PDF tables from real detector compare CSV.')
    p.add_argument('--input-csv', default='_runs/paper_data/real_detector_compare/real_world_results.csv')
    p.add_argument('--out-dir', default='_runs/paper_data/real_detector_compare/paper_tables')
    p.add_argument('--rows-per-page', type=int, default=18)
    args = p.parse_args()

    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f'Input CSV not found: {in_csv}')

    rows = _load_rows(in_csv)
    if not rows:
        raise ValueError(f'No rows parsed from: {in_csv}')

    overall = _overall_table(rows)
    by_cond_long = _by_condition_long_table(rows)
    paired_long, paired_compact = _paired_uv_vs_baseline(rows)
    pivot_target_conf, pivot_detect_rate = _by_condition_pivots(by_cond_long)
    pivot_delta_target_conf = _paired_pivot_delta_target(paired_compact)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}

    # CSVs (data provenance)
    csv_outputs = {
        'overall_summary_csv': out_dir / 'overall_summary.csv',
        'by_condition_long_csv': out_dir / 'by_condition_long.csv',
        'by_condition_pivot_target_conf_csv': out_dir / 'by_condition_pivot_target_conf.csv',
        'by_condition_pivot_detect_rate_csv': out_dir / 'by_condition_pivot_detect_rate.csv',
        'paired_uv_vs_baseline_long_csv': out_dir / 'paired_uv_vs_baseline_long.csv',
        'paired_uv_vs_baseline_compact_csv': out_dir / 'paired_uv_vs_baseline_compact.csv',
        'paired_uv_vs_baseline_pivot_delta_target_conf_csv': out_dir / 'paired_uv_vs_baseline_pivot_delta_target_conf.csv',
    }
    _write_csv(csv_outputs['overall_summary_csv'], overall)
    _write_csv(csv_outputs['by_condition_long_csv'], by_cond_long)
    _write_csv(csv_outputs['by_condition_pivot_target_conf_csv'], pivot_target_conf)
    _write_csv(csv_outputs['by_condition_pivot_detect_rate_csv'], pivot_detect_rate)
    _write_csv(csv_outputs['paired_uv_vs_baseline_long_csv'], paired_long)
    _write_csv(csv_outputs['paired_uv_vs_baseline_compact_csv'], paired_compact)
    _write_csv(csv_outputs['paired_uv_vs_baseline_pivot_delta_target_conf_csv'], pivot_delta_target_conf)

    # PDFs (paper-facing)
    pdf_outputs = {
        'overall_summary_pdf': out_dir / 'overall_summary.pdf',
        'by_condition_long_pdf': out_dir / 'by_condition_long.pdf',
        'by_condition_pivot_target_conf_pdf': out_dir / 'by_condition_pivot_target_conf.pdf',
        'by_condition_pivot_detect_rate_pdf': out_dir / 'by_condition_pivot_detect_rate.pdf',
        'paired_uv_vs_baseline_long_pdf': out_dir / 'paired_uv_vs_baseline_long.pdf',
        'paired_uv_vs_baseline_compact_pdf': out_dir / 'paired_uv_vs_baseline_compact.pdf',
        'paired_uv_vs_baseline_pivot_delta_target_conf_pdf': out_dir / 'paired_uv_vs_baseline_pivot_delta_target_conf.pdf',
    }

    _pdf_table(
        overall,
        pdf_outputs['overall_summary_pdf'],
        title='Real-World Detector Comparison (Overall)',
        subtitle='Mean ± std across all captured real-world images (45 images × 6 detectors rows in source CSV).',
        rows_per_page=int(args.rows_per_page),
    )
    _pdf_table(
        by_cond_long,
        pdf_outputs['by_condition_long_pdf'],
        title='Real-World Detector Comparison by Condition',
        subtitle='Compact metrics (mean ± std) across duplicate captures (_1/_2/_3).',
        rows_per_page=int(args.rows_per_page),
    )
    _pdf_table(
        pivot_target_conf,
        pdf_outputs['by_condition_pivot_target_conf_pdf'],
        title='Pivot Table: Target Confidence by Condition',
        subtitle='Rows=detectors, columns=distance|condition, values=mean ± std across duplicates.',
        rows_per_page=max(10, int(args.rows_per_page)),
    )
    _pdf_table(
        pivot_detect_rate,
        pdf_outputs['by_condition_pivot_detect_rate_pdf'],
        title='Pivot Table: Detect Rate by Condition',
        subtitle='Rows=detectors, columns=distance|condition, values=mean ± std across duplicates.',
        rows_per_page=max(10, int(args.rows_per_page)),
    )
    _pdf_table(
        paired_long,
        pdf_outputs['paired_uv_vs_baseline_long_pdf'],
        title='Paired UV vs Baseline (Per Duplicate Pair)',
        subtitle='Matched by detector + distance + ambient + headlights + replicate index.',
        rows_per_page=int(args.rows_per_page),
    )
    _pdf_table(
        paired_compact,
        pdf_outputs['paired_uv_vs_baseline_compact_pdf'],
        title='Paired UV vs Baseline (Compact)',
        subtitle='Mean ± std of UV-Baseline deltas across duplicate captures.',
        rows_per_page=int(args.rows_per_page),
    )
    _pdf_table(
        pivot_delta_target_conf,
        pdf_outputs['paired_uv_vs_baseline_pivot_delta_target_conf_pdf'],
        title='Pivot Table: Δ Target Confidence (UV - Baseline)',
        subtitle='Rows=detectors, columns=distance|ambient|headlights, values=mean ± std across duplicates.',
        rows_per_page=max(10, int(args.rows_per_page)),
    )

    for k, v in {**csv_outputs, **pdf_outputs}.items():
        outputs[k] = str(v)
        print(f"[SAVE] {v}")

    parse_summary = out_dir / 'parse_summary.json'
    _write_parse_summary(parse_summary, rows, outputs)
    print(f"[SAVE] {parse_summary}")
    print(f"[INFO] rows={len(rows)} parsed_ok={sum(1 for r in rows if r.get('parse_ok'))}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
