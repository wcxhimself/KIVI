import json
import os
import sys
import csv
from collections import defaultdict

EVAL_DIR = os.environ.get("KIVI_OUTPUT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

REQUIRED_FIELDS = [
    "factual_precision",
    "helpfulness_score",
    "verification_total_claims",
    "verification_correct",
    "verification_uncertain",
    "verification_incorrect",
    "category",
    "prompt",
]

STR_FIELDS = {"category", "prompt"}

MODEL_ORDER = [
    "seedance-2.0",
    "happyhorse-1.0",
    "wan2.2",
    "hunyuanvideo-1.5",
    "helios",
    "longcat",
    "longlive",
]


def sort_models(names):
    return sorted(names, key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else len(MODEL_ORDER))


def validate_score(score_path, data):
    for field in REQUIRED_FIELDS:
        val = data.get(field)
        if val is None:
            raise ValueError(
                f"Field '{field}' is None in {score_path}. "
                f"Ensure evaluation completed successfully."
            )
        if field in STR_FIELDS:
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"Field '{field}' is empty in {score_path} (value={val!r})."
                )


def load_helpfulness_subscores(eval_dir):
    hs_path = os.path.join(eval_dir, "helpfulness_score.json")
    if not os.path.isfile(hs_path):
        raise FileNotFoundError(f"Missing helpfulness_score.json in {eval_dir}")
    with open(hs_path) as f:
        hs_data = json.load(f)
    subscores = hs_data.get("subscores", {})
    rel = subscores.get("relevance")
    comp = subscores.get("completeness")
    clar = subscores.get("clarity")
    if rel is None or comp is None or clar is None:
        raise ValueError(
            f"Missing subscores in {hs_path}: "
            f"relevance={rel}, completeness={comp}, clarity={clar}"
        )
    return rel, comp, clar


def load_all_scores():
    records = []
    if not os.path.isdir(EVAL_DIR):
        print(f"Error: {EVAL_DIR} does not exist", file=sys.stderr)
        sys.exit(1)
    for model in sorted(os.listdir(EVAL_DIR)):
        model_dir = os.path.join(EVAL_DIR, model)
        if not os.path.isdir(model_dir) or model.startswith("common_") or model.startswith("ablation_"):
            continue
        for cat in sorted(os.listdir(model_dir)):
            cat_dir = os.path.join(model_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for qdir in sorted(os.listdir(cat_dir)):
                eval_dir = os.path.join(cat_dir, qdir, "evaluation")
                score_path = os.path.join(eval_dir, "score.json")
                if not os.path.isfile(score_path):
                    continue
                with open(score_path) as f:
                    data = json.load(f)
                validate_score(score_path, data)

                rel, comp, clar = load_helpfulness_subscores(eval_dir)
                data["_model"] = model
                data["_category"] = cat
                data["_prompt_dir"] = qdir
                data["_relevance"] = rel
                data["_completeness"] = comp
                data["_clarity"] = clar
                records.append(data)
    return records


def _key(r):
    return (r.get("category", ""), r.get("prompt", ""))


def sep(char="=", width=84):
    return char * width


PER_Q_WIDTH = 120
SUMMARY_WIDTH = 150


def print_comparison(records):
    by_q = defaultdict(list)
    for r in records:
        by_q[_key(r)].append(r)

    model_names = sort_models(set(r["_model"] for r in records))

    print(sep(width=PER_Q_WIDTH))
    print(f"{'MODEL COMPARISON PER PROMPT':^{PER_Q_WIDTH}}")
    print(sep(width=PER_Q_WIDTH))

    for qkey in sorted(by_q.keys()):
        cat, prompt = qkey
        rows = by_q[qkey]
        q_short = (prompt[:74] + "..") if len(prompt) > 74 else prompt
        print(f"\n{cat}  |  {q_short}")
        print("-" * PER_Q_WIDTH)

        hdr = f"{'Model':<20}"
        hdr += f"{'Fact%':>8}{'Help%':>8}{'Rel%':>8}{'Cmp%':>8}{'Clr%':>8}"
        hdr += f"{'Claims':>8}{'Corr':>8}{'Unc':>6}{'Wr':>6}"
        print(hdr)
        print("-" * len(hdr))

        for r in sorted(rows, key=lambda x: sort_models([x["_model"]])[0]):
            fp = r.get("factual_precision", "N/A")
            hs = r.get("helpfulness_score", "N/A")
            rel_val = (str(r["_relevance"] * 10) + "%") if isinstance(r.get("_relevance"), (int, float)) else "N/A"
            comp_val = (str(r["_completeness"] * 10) + "%") if isinstance(r.get("_completeness"), (int, float)) else "N/A"
            clar_val = (str(r["_clarity"] * 10) + "%") if isinstance(r.get("_clarity"), (int, float)) else "N/A"
            tc = r.get("verification_total_claims", "N/A")
            cc = r.get("verification_correct", "N/A")
            uc = r.get("verification_uncertain", "N/A")
            ic = r.get("verification_incorrect", "N/A")
            print(
                f"{r['_model']:<20}"
                f"{str(fp):>8}{str(hs):>8}{rel_val:>8}{comp_val:>8}{clar_val:>8}"
                f"{str(tc):>8}{str(cc):>8}{str(uc):>6}{str(ic):>6}"
            )

        if len(rows) > 1:
            avg_fp = sum(r.get("factual_precision", 0) or 0 for r in rows) / len(rows)
            avg_hs = sum(r.get("helpfulness_score", 0) or 0 for r in rows) / len(rows)
            avg_rel = sum(r.get("_relevance", 0) or 0 for r in rows) / len(rows) * 10
            avg_comp = sum(r.get("_completeness", 0) or 0 for r in rows) / len(rows) * 10
            avg_clar = sum(r.get("_clarity", 0) or 0 for r in rows) / len(rows) * 10
            print("-" * len(hdr))
            print(
                f"{'AVERAGE':<20}"
                f"{avg_fp:>8.2f}{avg_hs:>8.2f}{avg_rel:>8.1f}{avg_comp:>8.1f}{avg_clar:>8.1f}"
            )

    TABLE_WIDTH = 100
    by_cat_model = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_cat_model[r["_category"]][r["_model"]].append(r)

    print("\n\n")
    print(sep("=", width=TABLE_WIDTH))
    print(f"{'FACTUALITY BY CATEGORY (%)':^{TABLE_WIDTH}}")
    print(sep("=", width=TABLE_WIDTH))

    model_names_all = sort_models(set(r["_model"] for r in records))
    cats_all = sorted(by_cat_model.keys())
    cat_w = max(len(c) for c in cats_all) + 2
    col_w = 10
    hdr = f"{'Category':<{cat_w}}"
    for m in model_names_all:
        hdr += f"{m:>{20}}"
    print(hdr)
    print("-" * len(hdr))

    cat_fp = defaultdict(dict)
    for cat in cats_all:
        line = f"{cat:<{cat_w}}"
        for m in model_names_all:
            rows = by_cat_model[cat].get(m, [])
            if rows:
                avg = sum(r.get("factual_precision", 0) or 0 for r in rows) / len(rows)
                cat_fp[m][cat] = avg
                line += f"{avg:>{20}.2f}"
            else:
                cat_fp[m][cat] = None
                line += f"{'N/A':>{20}}"
        print(line)

    print("-" * len(hdr))
    line = f"{'AVERAGE':<{cat_w}}"
    for m in model_names_all:
        vals = [v for v in cat_fp[m].values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0
        line += f"{avg:>{20}.2f}"
    print(line)

    csv_dir = os.path.dirname(os.path.abspath(__file__))
    fp_csv = os.path.join(csv_dir, "factuality_by_category.csv")
    with open(fp_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category"] + model_names_all)
        for cat in cats_all:
            row = [cat] + [f"{cat_fp[m][cat]:.2f}" if cat_fp[m].get(cat) is not None else "N/A" for m in model_names_all]
            writer.writerow(row)
        avg_row = ["AVERAGE"]
        for m in model_names_all:
            vals = [v for v in cat_fp[m].values() if v is not None]
            avg_row.append(f"{sum(vals)/len(vals):.2f}" if vals else "N/A")
        writer.writerow(avg_row)
    print(f"Factuality CSV saved to: {fp_csv}")

    print("\n\n")
    print(sep("=", width=TABLE_WIDTH))
    print(f"{'HELPFULNESS BY CATEGORY (0-100)':^{TABLE_WIDTH}}")
    print(sep("=", width=TABLE_WIDTH))

    hdr = f"{'Category':<{cat_w}}"
    for m in model_names_all:
        hdr += f"{m:>{20}}"
    print(hdr)
    print("-" * len(hdr))

    cat_hs = defaultdict(dict)
    for cat in cats_all:
        line = f"{cat:<{cat_w}}"
        for m in model_names_all:
            rows = by_cat_model[cat].get(m, [])
            if rows:
                avg = sum(r.get("helpfulness_score", 0) or 0 for r in rows) / len(rows)
                cat_hs[m][cat] = avg
                line += f"{avg:>{20}.2f}"
            else:
                cat_hs[m][cat] = None
                line += f"{'N/A':>{20}}"
        print(line)

    print("-" * len(hdr))
    line = f"{'AVERAGE':<{cat_w}}"
    for m in model_names_all:
        vals = [v for v in cat_hs[m].values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0
        line += f"{avg:>{20}.2f}"
    print(line)

    hs_csv = os.path.join(csv_dir, "helpfulness_by_category.csv")
    with open(hs_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category"] + model_names_all)
        for cat in cats_all:
            row = [cat] + [f"{cat_hs[m][cat]:.2f}" if cat_hs[m].get(cat) is not None else "N/A" for m in model_names_all]
            writer.writerow(row)
        avg_row = ["AVERAGE"]
        for m in model_names_all:
            vals = [v for v in cat_hs[m].values() if v is not None]
            avg_row.append(f"{sum(vals)/len(vals):.2f}" if vals else "N/A")
        writer.writerow(avg_row)
    print(f"Helpfulness CSV saved to: {hs_csv}")

    print("\n\n")
    print(sep("=", width=SUMMARY_WIDTH))
    print(f"{'SUMMARY BY CATEGORY':^{SUMMARY_WIDTH}}")
    print(sep("=", width=SUMMARY_WIDTH))

    for cat in sorted(by_cat_model.keys()):
        print(f"\n{'─' * SUMMARY_WIDTH}")
        print(f"  Category: {cat}")
        print(f"{'─' * SUMMARY_WIDTH}")
        hdr = (
            f"{'Model':<20}{'QCnt':>6}{'AvgFact%':>10}{'AvgHelp%':>10}"
            f"{'AvgRel':>8}{'AvgCmp':>8}{'AvgClr':>8}"
            f"{'TotClaims':>10}{'TotCorr':>10}{'TotUnc':>10}{'TotWr':>10}"
        )
        print(hdr)
        print("-" * len(hdr))

        for model in sort_models(by_cat_model[cat].keys()):
            rows = by_cat_model[cat][model]
            qc = len(rows)
            avg_fp = sum(r.get("factual_precision", 0) or 0 for r in rows) / qc
            avg_hs = sum(r.get("helpfulness_score", 0) or 0 for r in rows) / qc
            avg_rel = sum(r["_relevance"] for r in rows) / qc
            avg_comp = sum(r["_completeness"] for r in rows) / qc
            avg_clar = sum(r["_clarity"] for r in rows) / qc
            tot_c = sum(r.get("verification_total_claims", 0) or 0 for r in rows)
            tot_cor = sum(r.get("verification_correct", 0) or 0 for r in rows)
            tot_unc = sum(r.get("verification_uncertain", 0) or 0 for r in rows)
            tot_wro = sum(r.get("verification_incorrect", 0) or 0 for r in rows)
            print(
                f"{model:<20}"
                f"{qc:>6}{avg_fp:>10.2f}{avg_hs:>10.2f}"
                f"{avg_rel:>8.1f}{avg_comp:>8.1f}{avg_clar:>8.1f}"
                f"{tot_c:>10}{tot_cor:>10}{tot_unc:>10}{tot_wro:>10}"
            )

    print("\n\n")
    print(sep("=", width=SUMMARY_WIDTH))
    print(f"{'OVERALL SUMMARY':^{SUMMARY_WIDTH}}")
    print(sep("=", width=SUMMARY_WIDTH))
    hdr = (
        f"{'Model':<20}{'QCnt':>6}{'AvgFact%':>10}{'AvgHelp%':>10}"
        f"{'AvgRel':>8}{'AvgCmp':>8}{'AvgClr':>8}"
        f"{'TotClaims':>10}{'TotCorr':>10}{'TotUnc':>10}{'TotWr':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    by_model = defaultdict(list)
    for r in records:
        by_model[r["_model"]].append(r)

    for model in sort_models(by_model.keys()):
        rows = by_model[model]
        qc = len(rows)
        avg_fp = sum(r.get("factual_precision", 0) or 0 for r in rows) / qc
        avg_hs = sum(r.get("helpfulness_score", 0) or 0 for r in rows) / qc
        avg_rel = sum(r["_relevance"] for r in rows) / qc
        avg_comp = sum(r["_completeness"] for r in rows) / qc
        avg_clar = sum(r["_clarity"] for r in rows) / qc
        tot_c = sum(r.get("verification_total_claims", 0) or 0 for r in rows)
        tot_cor = sum(r.get("verification_correct", 0) or 0 for r in rows)
        tot_unc = sum(r.get("verification_uncertain", 0) or 0 for r in rows)
        tot_wro = sum(r.get("verification_incorrect", 0) or 0 for r in rows)
        print(
            f"{model:<20}"
            f"{qc:>6}{avg_fp:>10.2f}{avg_hs:>10.2f}"
            f"{avg_rel:>8.1f}{avg_comp:>8.1f}{avg_clar:>8.1f}"
            f"{tot_c:>10}{tot_cor:>10}{tot_unc:>10}{tot_wro:>10}"
        )

    HELPF_WIDTH = 80
    print("\n\n")
    print(sep("=", width=HELPF_WIDTH))
    print(f"{'HELPFULNESS DIMENSIONS (% , 0-100)':^{HELPF_WIDTH}}")
    print(sep("=", width=HELPF_WIDTH))
    hdr = f"{'Model':<20}{'QCnt':>6}{'relevance':>12}{'completeness':>14}{'clarity':>10}"
    print(hdr)
    print("-" * len(hdr))

    for model in sort_models(by_model.keys()):
        rows = by_model[model]
        qc = len(rows)
        avg_rel = sum(r["_relevance"] for r in rows) / qc * 10
        avg_comp = sum(r["_completeness"] for r in rows) / qc * 10
        avg_clar = sum(r["_clarity"] for r in rows) / qc * 10
        print(
            f"{model:<20}"
            f"{qc:>6}{avg_rel:>12.1f}{avg_comp:>14.1f}{avg_clar:>10.1f}"
        )

    print("\n\n")
    print(sep("=", width=SUMMARY_WIDTH))
    print(f"{'PAIRWISE COMPARISON':^{SUMMARY_WIDTH}}")
    print(sep("=", width=SUMMARY_WIDTH))

    q_to_models = defaultdict(set)
    for r in records:
        q_to_models[_key(r)].add(r["_model"])

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            shared = []
            for qk, ms in q_to_models.items():
                if m1 in ms and m2 in ms:
                    r1 = [r for r in records if _key(r) == qk and r["_model"] == m1][0]
                    r2 = [r for r in records if _key(r) == qk and r["_model"] == m2][0]
                    shared.append((r1, r2))

            if not shared:
                continue

            print(f"\n  {m1}  vs  {m2}  ({len(shared)} shared prompts)")
            print(f"  {'─' * 80}")
            fp1 = [s[0].get("factual_precision", 0) or 0 for s in shared]
            fp2 = [s[1].get("factual_precision", 0) or 0 for s in shared]
            hs1 = [s[0].get("helpfulness_score", 0) or 0 for s in shared]
            hs2 = [s[1].get("helpfulness_score", 0) or 0 for s in shared]
            rel1 = [s[0]["_relevance"] for s in shared]
            rel2 = [s[1]["_relevance"] for s in shared]
            comp1 = [s[0]["_completeness"] for s in shared]
            comp2 = [s[1]["_completeness"] for s in shared]
            clar1 = [s[0]["_clarity"] for s in shared]
            clar2 = [s[1]["_clarity"] for s in shared]

            print(f"    Avg Fact%:    {m1}={sum(fp1)/len(fp1):.2f}%  vs  {m2}={sum(fp2)/len(fp2):.2f}%")
            print(f"    Avg Help%:    {m1}={sum(hs1)/len(hs1):.2f}  vs  {m2}={sum(hs2)/len(hs2):.2f}")
            print(f"    Dims: rel {m1}={sum(rel1)/len(rel1)*10:.1f}% vs {m2}={sum(rel2)/len(rel2)*10:.1f}%  "
                  f"cmp {m1}={sum(comp1)/len(comp1)*10:.1f}% vs {m2}={sum(comp2)/len(comp2)*10:.1f}%  "
                  f"clr {m1}={sum(clar1)/len(clar1)*10:.1f}% vs {m2}={sum(clar2)/len(clar2)*10:.1f}%")

            wins_fp = sum(1 for a, b in zip(fp1, fp2) if a > b)
            wins_hs = sum(1 for a, b in zip(hs1, hs2) if a > b)
            ties_fp = sum(1 for a, b in zip(fp1, fp2) if a == b)
            ties_hs = sum(1 for a, b in zip(hs1, hs2) if a == b)
            print(f"    Fact% wins: {m1}={wins_fp}, {m2}={len(shared)-wins_fp-ties_fp}, ties={ties_fp}")
            print(f"    Help% wins: {m1}={wins_hs}, {m2}={len(shared)-wins_hs-ties_hs}, ties={ties_hs}")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    records = load_all_scores()
    if not records:
        print("No score.json files found.")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "compare_eval_results.txt")
    with open(output_path, "w", encoding="utf-8") as log:
        old_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log)
        try:
            print_comparison(records)
        finally:
            sys.stdout = old_stdout
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()