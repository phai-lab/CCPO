"""Checks of the context features, peer readouts, credits and normalization."""
from dataclasses import fields, replace
import unittest

import numpy as np

from ccpo import CCPOConfig, RolloutBatch, compute_ccpo_advantage, context_features, trajectory_statistics


def features(rows, width):
    return np.sin(np.arange(rows * width).reshape(rows, width) ** 2 * 0.37 + 0.5)


def fixture():
    lengths = np.repeat([4, 4, 3, 3], [4, 4, 3, 3])
    steps = np.concatenate([np.arange(n) for n in [4, 4, 3, 3]])
    outcomes = np.repeat([10., 0., 10., 0.], [4, 4, 3, 3])
    penalties = np.zeros(len(steps))
    penalties[1] = .1
    return RolloutBatch(
        task_ids=np.array(['task'] * len(steps)),
        trajectory_ids=np.repeat(['a', 'b', 'c', 'd'], [4, 4, 3, 3]),
        observations=np.array(['room-' + str(t % 2) for t in steps]),
        steps=steps, lengths=lengths, hidden=features(len(steps), 8),
        returns=.95 ** (lengths - steps - 1) * outcomes - penalties,
        outcomes=outcomes, episode_scores=outcomes - penalties,
    )


def take(batch, rows):
    return RolloutBatch(**{field.name: np.asarray(getattr(batch, field.name))[rows] for field in fields(batch)})


def standardize(values):
    return (values - values.mean()) / (values.std(ddof=1) + 1e-6)


class ContextTests(unittest.TestCase):
    def test_statistics_use_the_prefix_and_exact_observation_text(self):
        obs = np.array(['A', 'B', 'A', 'C', 'a'])
        steps = np.arange(5)
        order = np.array([3, 0, 4, 2, 1])
        actual = trajectory_statistics(obs[order], ['traj'] * 5, steps[order])
        expected = np.array([[0, 1, 0, 1], [1, 2, 0, 1], [2, 2, 1, 2/3],
                             [3, 3, 0, 3/4], [4, 4, 0, 4/5]])
        np.testing.assert_allclose(actual, expected[order])

    def test_summary_encoding_and_equal_block_weights(self):
        hidden = features(8, 16)
        statistics = np.tile([15, 12, 1, .5], (8, 1))
        phi = context_features(hidden, statistics)
        self.assertEqual(phi.shape, (8, 16 + 37))
        np.testing.assert_allclose(np.linalg.norm(phi, axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(phi[:, :16], axis=1), 1/np.sqrt(2))
        np.testing.assert_allclose(np.linalg.norm(phi[:, 16:], axis=1), 1/np.sqrt(2))
        expected = np.r_[np.ones(6), np.zeros(6), np.ones(5), np.zeros(7), np.ones(6), np.zeros(6), 1]
        np.testing.assert_allclose(phi[0, 16:], expected / np.linalg.norm(expected) / np.sqrt(2))
        a = context_features(hidden, np.tile([30, 25, 1, 1], (8, 1)))
        b = context_features(hidden, np.tile([80, 70, 1, 1], (8, 1)))
        np.testing.assert_array_equal(a, b)

    def test_hidden_states_are_centered_over_the_batch(self):
        batch = fixture()
        hidden = batch.hidden.copy()
        statistics = trajectory_statistics(batch.observations, batch.trajectory_ids, batch.steps)
        a = context_features(hidden, statistics)
        b = context_features(hidden + np.arange(8), statistics)
        np.testing.assert_allclose(a, b, atol=1e-13)
        np.testing.assert_array_equal(hidden, batch.hidden)
        with self.assertRaises(ValueError):
            context_features(hidden * np.nan, statistics)


class CreditTests(unittest.TestCase):
    def test_two_trajectory_baselines_and_terminal_values(self):
        batch = RolloutBatch(
            task_ids=['q'] * 6, trajectory_ids=['a'] * 3 + ['b'] * 3,
            observations=['x', 'y', 'z'] * 2, steps=np.tile(np.arange(3), 2),
            lengths=np.full(6, 3), hidden=np.zeros((6, 4)),
            returns=np.array([9.025, 9.5, 10., 0., 0., 0.]),
            outcomes=np.array([10.] * 3 + [0.] * 3),
            episode_scores=np.array([10.] * 3 + [0.] * 3),
        )
        result = compute_ccpo_advantage(batch, CCPOConfig(benchmark='webshop'))
        np.testing.assert_allclose(result.baseline, [0, 0, 0, 9.025, 9.5, 10])
        np.testing.assert_allclose(result.value, [0, 0, 0, 10*.95**3, 10*.95**2, 9.5])
        np.testing.assert_allclose(result.endpoint_value, [0, 10, 10, 9.5, 0, 0])
        np.testing.assert_allclose(result.historical, batch.returns - result.baseline)
        self.assertFalse(result.used_fallback.any())

    def test_query_trajectory_is_excluded_from_its_own_peers(self):
        batch = fixture()
        own = batch.trajectory_ids == 'a'
        returns = batch.returns.copy()
        returns[own] += 6
        outcomes = batch.outcomes.copy()
        outcomes[own] = 0
        before = compute_ccpo_advantage(batch)
        after = compute_ccpo_advantage(replace(batch, returns=returns, outcomes=outcomes))
        np.testing.assert_array_equal(before.baseline[own], after.baseline[own])
        np.testing.assert_array_equal(before.value[own], after.value[own])
        self.assertGreater(np.max(abs(before.value[~own] - after.value[~own])), .1)

    def test_task_fallback_and_tasks_without_peers(self):
        batch = fixture()
        unique = replace(batch, observations=np.array(['unique-' + str(i) for i in range(len(batch.steps))]))
        result = compute_ccpo_advantage(unique)
        self.assertTrue(result.used_fallback.all())
        self.assertTrue(result.supported.all())
        isolated = replace(unique, task_ids=unique.trajectory_ids.copy())
        result = compute_ccpo_advantage(isolated)
        np.testing.assert_array_equal(result.advantages, 0)
        self.assertFalse(result.supported.any())
        self.assertTrue(np.isnan(result.value).all())

    def test_future_value_change_telescopes_over_the_horizon(self):
        batch = fixture()
        one = compute_ccpo_advantage(batch, CCPOConfig(future_horizon=1))
        two = compute_ccpo_advantage(batch, CCPOConfig(future_horizon=2))
        delta_one = one.endpoint_value - one.value
        expected = delta_one.copy()
        for i, step in enumerate(batch.steps):
            if step + 1 < batch.lengths[i]:
                expected[i] += delta_one[i + 1]
        np.testing.assert_allclose(two.endpoint_value - two.value, expected, atol=1e-14)
        np.testing.assert_allclose(two.future, standardize(expected), atol=1e-14)

    def test_episode_variant_and_benchmark_normalization(self):
        batch = fixture()
        for benchmark in ['alfworld', 'webshop']:
            off = compute_ccpo_advantage(batch, CCPOConfig(benchmark=benchmark))
            on = compute_ccpo_advantage(batch, CCPOConfig(benchmark=benchmark, episode_weight=1))
            episode = batch.episode_scores - batch.episode_scores.mean()
            contextual = off.historical + off.future
            if benchmark == 'alfworld':
                episode /= batch.episode_scores.std(ddof=1) + 1e-6
                contextual = standardize(contextual)
            np.testing.assert_allclose(on.episode, episode, atol=1e-14)
            np.testing.assert_allclose(off.advantages, contextual, atol=1e-14)
            np.testing.assert_allclose(on.advantages, contextual + episode, atol=1e-14)
            np.testing.assert_array_equal(on.contextual, off.contextual)

    def test_episode_scores_do_not_affect_ccpo_without_the_episode_channel(self):
        batch = fixture()
        alternate = replace(batch, episode_scores=np.arange(len(batch.steps)) ** 2)
        for benchmark in ['alfworld', 'webshop']:
            config = CCPOConfig(benchmark=benchmark)
            a, b = compute_ccpo_advantage(batch, config), compute_ccpo_advantage(alternate, config)
            np.testing.assert_array_equal(a.advantages, b.advantages)
            self.assertFalse(np.allclose(a.episode, b.episode))

    def test_row_order_and_duplicate_rows_do_not_change_credit(self):
        batch = fixture()
        order = np.array([9, 0, 13, 2, 7, 11, 1, 4, 12, 6, 3, 10, 8, 5, 0, 0, 5])
        for benchmark in ['alfworld', 'webshop']:
            config = CCPOConfig(benchmark=benchmark)
            base = compute_ccpo_advantage(batch, config)
            shuffled = compute_ccpo_advantage(take(batch, order), config)
            for name in ['historical', 'future', 'baseline', 'value', 'endpoint_value']:
                np.testing.assert_allclose(getattr(shuffled, name), getattr(base, name)[order], atol=1e-13)
            expected = (base.historical + base.future)[order]
            if benchmark == 'alfworld':
                expected = standardize(expected)
            np.testing.assert_allclose(shuffled.advantages, expected, atol=1e-13)

    def test_single_row_task(self):
        batch = RolloutBatch(['q'], ['a'], ['obs'], np.array([0]), np.array([1]),
                             np.ones((1, 4)), np.array([10.]), np.array([10.]), np.array([10.]))
        for benchmark, expected in [('alfworld', 10/(1+1e-6)), ('webshop', 10.)]:
            result = compute_ccpo_advantage(batch, CCPOConfig(benchmark=benchmark, episode_weight=1))
            self.assertAlmostEqual(result.advantages[0], expected)
            self.assertEqual(result.contextual[0], 0.)

    def test_invalid_batches_are_rejected(self):
        batch = fixture()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            compute_ccpo_advantage(take(batch, np.arange(1, len(batch.steps))))
        padded = take(batch, np.r_[np.arange(len(batch.steps)), 0])
        bad = padded.hidden.copy()
        bad[-1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'disagree'):
            compute_ccpo_advantage(replace(padded, hidden=bad))
        bad[-1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            compute_ccpo_advantage(replace(padded, hidden=bad))
        tasks = batch.task_ids.copy()
        tasks[0] = 'other'
        with self.assertRaisesRegex(ValueError, 'unique across tasks'):
            compute_ccpo_advantage(replace(batch, task_ids=tasks))
        outcomes = batch.outcomes.copy()
        outcomes[0] = 0
        with self.assertRaisesRegex(ValueError, 'constant'):
            compute_ccpo_advantage(replace(batch, outcomes=outcomes))
        with self.assertRaises(ValueError):
            compute_ccpo_advantage(replace(batch, steps=batch.steps.astype(float)))

    def test_invalid_configurations_are_rejected(self):
        for change in [dict(benchmark='other'), dict(episode_weight=-1), dict(episode_weight=True),
                       dict(future_horizon=0), dict(future_horizon=True), dict(future_horizon=1.5),
                       dict(gamma=np.nan), dict(gamma=1.1), dict(bandwidth_scale=0),
                       dict(num_removed_pcs=-1)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                CCPOConfig(**change)


if __name__ == '__main__':
    unittest.main()
