"""Checks of the verl-agent adapter: hidden-state extraction and token advantages."""
from types import SimpleNamespace
import unittest

import numpy as np

from ccpo import CCPOConfig
from test_ccpo import fixture

try:
    import torch
    from ccpo.verl_adapter import HIDDEN_KEY, STEP_KEY, compute_verl_advantages, last_prompt_hidden
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install ccpo[torch] to test the verl-agent adapter')
class VerlAdapterTests(unittest.TestCase):
    def test_padded_and_packed_inputs_select_the_same_prompt_token(self):
        mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]])
        hidden = torch.arange(2*7*4, dtype=torch.float32).reshape(2, 7, 4).requires_grad_()
        indices = mask.reshape(-1).nonzero().flatten()
        packed = hidden.reshape(-1, 4)[indices][None]
        a = last_prompt_hidden(hidden, mask, response_length=3)
        b = last_prompt_hidden(packed, mask, response_length=3, packed_indices=indices)
        torch.testing.assert_close(a, hidden[:, 3].detach())
        torch.testing.assert_close(a, b)
        self.assertFalse(a.requires_grad)
        with self.assertRaises(ValueError):
            last_prompt_hidden(packed, mask, 3, packed_indices=indices.flip(0))
        with self.assertRaises(ValueError):
            last_prompt_hidden(hidden * float('nan'), mask, 3)

    def data(self):
        batch = fixture()
        n = len(batch.steps)
        mask = torch.ones(n, 4)
        mask[::2, -1] = 0
        rewards = torch.zeros_like(mask)
        rewards[torch.arange(n), mask.sum(-1).long()-1] = torch.tensor(batch.episode_scores, dtype=torch.float32)
        data = SimpleNamespace(
            batch={
                'response_mask': mask, 'token_level_rewards': rewards.requires_grad_(),
                'step_rewards': torch.tensor(batch.returns, requires_grad=True),
                HIDDEN_KEY: torch.tensor(batch.hidden, requires_grad=True),
            },
            non_tensor_batch={
                'uid': batch.task_ids, 'traj_uid': batch.trajectory_ids,
                'anchor_obs': batch.observations, STEP_KEY: batch.steps,
                'episode_lengths': batch.lengths, 'episode_rewards': batch.outcomes,
            },
        )
        # verl-agent collates metadata into object arrays, with float episode lengths.
        data.non_tensor_batch = {key: np.asarray(value, dtype=object)
                                 for key, value in data.non_tensor_batch.items()}
        data.non_tensor_batch['episode_lengths'] = batch.lengths.astype(np.float32).astype(object)
        return data

    def test_advantages_cover_response_tokens_and_carry_no_gradient(self):
        for benchmark in ['alfworld', 'webshop']:
            gradients, outputs = [], []
            for weight in [0, 1]:
                data = self.data()
                mask = data.batch['response_mask']
                advantages, result = compute_verl_advantages(data, CCPOConfig(
                    benchmark=benchmark, episode_weight=weight))
                self.assertFalse(advantages.requires_grad)
                self.assertEqual(advantages.dtype, torch.float32)
                torch.testing.assert_close(advantages[mask == 0], torch.zeros_like(advantages[mask == 0]))
                np.testing.assert_allclose(advantages.numpy(), result.advantages[:, None] * mask.numpy(),
                                           rtol=1e-6, atol=1e-6)
                logits = torch.linspace(-1, 1, mask.numel()*2).reshape(*mask.shape, 2).requires_grad_()
                logp = logits.log_softmax(-1)[..., 0]
                ratio = (logp - logp.detach()).exp()
                loss = -(torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages) * mask).sum() / mask.sum()
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(float(logits.grad[:, 0].abs().sum()), 0.)
                for name in ['step_rewards', 'token_level_rewards', HIDDEN_KEY]:
                    self.assertIsNone(data.batch[name].grad)
                gradients.append(logits.grad)
                outputs.append((advantages, result))
            self.assertFalse(torch.allclose(gradients[0], gradients[1]))
            expected = torch.tensor(outputs[1][1].episode, dtype=torch.float32)[:, None] * mask
            torch.testing.assert_close(outputs[1][0] - outputs[0][0], expected, atol=2e-6, rtol=1e-6)

    def test_step_metadata_must_be_integral(self):
        for key in ('episode_lengths', STEP_KEY):
            for value in (1.5, float('nan'), float('inf'), True):
                with self.subTest(key=key, value=value):
                    data = self.data()
                    data.non_tensor_batch[key][0] = value
                    with self.assertRaisesRegex(ValueError, 'integral metadata'):
                        compute_verl_advantages(data)

    def test_rewards_on_padding_and_nonbinary_masks_are_rejected(self):
        data = self.data()
        data.batch['token_level_rewards'] = data.batch['token_level_rewards'].detach()
        data.batch['token_level_rewards'][0, -1] = 1
        with self.assertRaisesRegex(ValueError, 'Masked'):
            compute_verl_advantages(data)
        data = self.data()
        data.batch['response_mask'][0, 0] = .5
        with self.assertRaisesRegex(ValueError, 'binary'):
            compute_verl_advantages(data)


if __name__ == '__main__':
    unittest.main()
