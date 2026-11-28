#!/usr/bin/env python3
"""Reproduce QFrBLiMa Table 5 from saved original and inverted judgments.
Run from repository root. Uses Python standard library only.
"""
import argparse
import csv
import re
import unicodedata
from pathlib import Path

MODELS = ('gemini', 'claude', 'gpt54', 'gemma4_12b_it', 'mistral_nemo_12b', 'qwen3_8b', 'aya_expanse_8b', 'llama31_8b')
LABELS = {'A_MUCH_MORE_FQ', 'A_SLIGHTLY_MORE_FQ', 'B_MUCH_MORE_FQ', 'B_SLIGHTLY_MORE_FQ', 'EQUIVALENT'}

def normalize(text):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold()))

def contains(needle, haystack):
    return bool(needle) and f' {needle} ' in f' {haystack} '

def score(row):
    q, i, a, b, mq, mi = map(normalize, (row['fq_evidence_span'], row['fi_evidence_span'], row['candidate_a'], row['candidate_b'], row['marker_FQ'], row['marker_FI']))
    label = row['overall_preference']
    pos = row['true_fq_position']
    if label not in LABELS or pos not in ('A', 'B'):
        raise ValueError(f'Invalid label or position: {row}')
    equivalent = label == 'EQUIVALENT'
    correct = not equivalent and label.startswith(pos + '_')
    if row['correct'].lower() != str(correct).lower():
        raise ValueError(f'Correctness mismatch: {row["judgment_id"]}')
    fq, fi = (a, b) if pos == 'A' else (b, a)
    grounded = not equivalent and ((contains(q, a) and contains(i, b)) or (contains(q, b) and contains(i, a)))
    attributed = not equivalent and contains(mq, q) and contains(mi, i) and contains(q, fq) and contains(i, fi)
    reversed_match = not equivalent and contains(mi, q) and contains(mq, i) and contains(q, fi) and contains(i, fq)
    identified = attributed or reversed_match
    assert not attributed or identified
    assert not identified or grounded
    return dict(grounded=grounded, identified=identified, attributed=attributed, correct=correct, equivalent=equivalent)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def pct(n, d):
    return f'{100*n/d:.2f}' if d else ''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, default=Path('data/qfrblima.csv'))
    p.add_argument('--predictions', type=Path, default=Path('results/predictions'))
    p.add_argument('--out', type=Path, default=Path('results/analysis/evidence_spans'))
    args = p.parse_args()
    benchmark = read_csv(args.benchmark)
    pairs = {r['pair_id']: r for r in benchmark}
    if len(pairs) != len(benchmark): raise ValueError('Duplicate benchmark pair IDs')
    summary, details = [], []
    for model in MODELS:
        model_rows = []
        positions = {}
        for order in ('original', 'inverted'):
            rows = read_csv(args.predictions / model / order / 'results.csv')
            ids = [r['pair_id'] for r in rows]
            if len(ids) != len(pairs) or set(ids) != set(pairs):
                raise ValueError(f'Incomplete or duplicate pair IDs: {model}/{order}')
            for r in rows:
                pair = pairs[r['pair_id']]
                pos = r['true_fq_position']
                if pos not in ('A','B'): raise ValueError('Invalid position')
                fq, fi = (r['candidate_a'], r['candidate_b']) if pos == 'A' else (r['candidate_b'], r['candidate_a'])
                if any(normalize(x) != normalize(y) for x,y in ((fq,pair['text_FQ']), (fi,pair['text_FI']), (r['marker_FQ'],pair['marker_FQ']), (r['marker_FI'],pair['marker_FI']))):
                    raise ValueError(f'Benchmark mismatch: {model}/{order}/{r["pair_id"]}')
                if order == 'original': positions[r['pair_id']] = pos
                elif positions[r['pair_id']] == pos: raise ValueError('Order not inverted')
                s = score(r)
                detail = dict(model=model, order=order, pair_id=r['pair_id'], judgment_id=r['judgment_id'], **s)
                model_rows.append(detail); details.append(detail)
        n = len(model_rows)
        g = sum(r['grounded'] for r in model_rows)
        i = sum(r['identified'] for r in model_rows)
        a = sum(r['attributed'] for r in model_rows)
        c = sum(r['identified'] and r['correct'] for r in model_rows)
        wrong = sum(not r['correct'] for r in model_rows)
        iw = sum(r['identified'] and not r['correct'] for r in model_rows)
        ne = sum(not r['equivalent'] for r in model_rows)
        ine = sum(r['identified'] and not r['equivalent'] for r in model_rows)
        aw = sum(r['attributed'] and not r['correct'] for r in model_rows)
        summary.append(dict(model=model, judgments=n, grounded=g, identified=i, attributed=a, correct_given_identified=c, grounding_pct=pct(g,n), identification_pct=pct(i,n), attribution_pct=pct(a,n), conditional_accuracy_pct=pct(c,i), incorrect=wrong, identified_incorrect=iw, identification_given_incorrect_pct=pct(iw,wrong), non_equivalent=ne, identified_non_equivalent=ine, identification_given_non_equivalent_pct=pct(ine,ne), attributed_incorrect=aw))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'evidence_span_metrics.csv', summary)
    write_csv(args.out/'evidence_span_judgments.csv', details)
    for r in summary:
        print(r['model'], *(r[k] for k in ('grounding_pct','identification_pct','attribution_pct','conditional_accuracy_pct')))
    print(f'Analyzed {len(details)} judgments; outputs: {args.out}')

if __name__ == '__main__':
    main()
