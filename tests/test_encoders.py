"""CPU-only regression tests for the released encoder experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(os.environ.get('QFR_TEST_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'scripts' / 'encoders'))
import analyze_classifiers as analysis
import compare_models as comparison


class StatisticalTests(unittest.TestCase):
    def test_holm_unsorted_and_capped(self):
        np.testing.assert_allclose(analysis.holm([.04, .01, .03, .9]), [.09, .04, .09, .9])
        np.testing.assert_allclose(analysis.holm([.8, .7]), [1, 1])

    def test_bootstrap_reproducibility(self):
        x = np.array([0., .2, .8, 1.])
        np.testing.assert_array_equal(analysis.ci(x, 1000, 42), analysis.ci(x, 1000, 42))
        np.testing.assert_allclose(analysis.ci(x, 1000, 42), comparison.interval(x, 1000, 42))
        np.testing.assert_allclose(analysis.ci(np.ones(5), 100, 42), [1, 1])

    def test_permutation_reproducibility(self):
        x = np.array([1., -1., .2, 0.])
        def draw():
            return analysis.resample(x, 1025, np.random.default_rng(42), permutation=True)
        np.testing.assert_array_equal(draw(), draw())
        self.assertEqual(len(draw()), 1025)
        np.testing.assert_array_equal(analysis.resample(np.zeros(4), 10, np.random.default_rng(42), permutation=True), np.zeros(10))

    def test_duplicate_pair_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.csv'
            path.write_text('pair_id,feature_id\np1,m1\np1,m1\n')
            with self.assertRaisesRegex(ValueError, 'duplicate pair IDs'):
                comparison.load(path)

    def test_missing_pair_rejected(self):
        b = pd.DataFrame(index=['p1', 'p2'])
        with self.assertRaisesRegex(ValueError, 'pair IDs differ'):
            comparison.align(b.iloc[:1], b, 'fixture')

    def test_invalid_boolean_rejected(self):
        with self.assertRaises(ValueError):
            comparison.bool_col(pd.DataFrame({'flag': ['true', '']}), 'flag')

    def test_strict_ordering_ties_and_threshold(self):
        # Exercise the same decision rules used by the analysis scripts.
        fi = pd.Series([.2, .5, .8, .4])
        fq = pd.Series([.3, .5, .7, .5])
        np.testing.assert_array_equal(fq.gt(fi), [True, False, False, True])
        np.testing.assert_array_equal((fi < .5) & (fq >= .5), [False, False, False, True])


class ReleasedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = ROOT / 'results/encoders/evaluation'
        cls.training = ROOT / 'results/encoders/training'

    def test_training_configurations_and_checkpoints(self):
        expected = dict(epochs=3, batch_size=16, grad_accum=2, eval_batch_size=64,
                        max_length=128, lr=2e-5, weight_decay=.01, warmup_ratio=.06,
                        seed=42, workers=4, torch='2.7.1')
        for model in analysis.MODELS:
            with self.subTest(model=model):
                folder = self.training / model
                cfg = analysis.read_json(folder / 'run_config.json')
                self.assertEqual({k: cfg[k] for k in expected}, expected)
                history = pd.read_csv(folder / 'history.csv')
                val = analysis.read_json(folder / 'val_metrics.json')
                test = analysis.read_json(folder / 'test_metrics.json')
                best = history.loc[history.val_f1_macro.idxmax()]
                self.assertEqual(int(best.epoch), val['best_epoch'])
                self.assertEqual(val['best_epoch'], test['best_epoch'])
                self.assertAlmostEqual(best.val_f1_macro, val['f1_macro'], places=8)
                for metrics, n in [(val, 93768), (test, 93428)]:
                    cm = np.asarray(metrics['confusion_matrix_rows_true_0_1'])
                    self.assertEqual(metrics['n'], n)
                    np.testing.assert_array_equal(cm.sum(axis=1), [n//2, n//2])
                    self.assertAlmostEqual(np.trace(cm)/n, metrics['accuracy'])

    def test_prediction_metrics(self):
        for model in analysis.MODELS:
            with self.subTest(model=model):
                folder = self.evaluation / model
                d = comparison.load(folder / 'paired_predictions.csv')
                m = analysis.read_json(folder / 'overall_metrics.json')
                self.assertEqual(len(d), 1205)
                counts = d.groupby('feature_id').size()
                self.assertEqual(len(counts), 241)
                self.assertTrue(counts.eq(5).all())
                fi, fq = d.score_fq_FI.astype(float), d.score_fq_FQ.astype(float)
                self.assertTrue(fi.between(0, 1).all() and fq.between(0, 1).all())
                strict = fq.gt(fi)
                both = (fi < .5) & (fq >= .5)
                np.testing.assert_array_equal(strict, comparison.bool_col(d, 'pairwise_success'))
                np.testing.assert_array_equal(both, comparison.bool_col(d, 'both_correct'))
                self.assertAlmostEqual(strict.mean(), m['pairwise_FQ_higher'])
                self.assertAlmostEqual(both.mean(), m['both_correct'])
                self.assertAlmostEqual(fi.eq(fq).mean(), m['pairwise_ties'])
                y = np.repeat([0, 1], len(d))
                pred = np.r_[fi >= .5, fq >= .5].astype(int)
                np.testing.assert_array_equal(confusion_matrix(y, pred, labels=[0,1]), m['confusion_matrix_FI_FQ'])
                self.assertAlmostEqual((y == pred).mean(), m['accuracy'])
                self.assertAlmostEqual(f1_score(y, pred, average='macro'), m['f1_macro'])
                marker = strict.groupby(d.feature_id).mean()
                self.assertEqual(dict(full=int(marker.eq(1).sum()), partial=int(((marker > 0)&(marker < 1)).sum()), none=int(marker.eq(0).sum())), m['marker_recognition'])
                np.testing.assert_allclose(analysis.ci(marker.to_numpy(), m['bootstrap_replicates'], m['seed']), m['pairwise_marker_cluster_95ci'])

    def test_original_benchmark_alignment_and_hash(self):
        path = ROOT / 'data/qfrblima.csv'
        b = comparison.load(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for model in analysis.MODELS:
            with self.subTest(model=model):
                folder = self.evaluation / model
                d = comparison.load(folder / 'paired_predictions.csv')
                comparison.align(d, b, model)
                for col in ('text_FI', 'text_FQ'):
                    self.assertTrue(d[col].equals(b[col]))
                self.assertEqual(digest, analysis.read_json(folder / 'overall_metrics.json')['benchmark_sha256'])


if __name__ == '__main__':
    unittest.main()
