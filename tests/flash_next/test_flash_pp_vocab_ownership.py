import copy
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from flash_pp_vocab_ownership import FLAG, build_pp_vocab_module, should_omit


def fixture(rank=0):
    config = NS(
        model_config=NS(
            hf_config=NS(model_type="qwen4_exp", tie_word_embeddings=False),
            hf_text_config=NS(
                tie_word_embeddings=False,
                num_hidden_layers=48,
                hidden_size=2560,
                vocab_size=248320,
            ),
            multimodal_config=NS(language_model_only=True),
        ),
        speculative_config=None,
        lora_config=None,
        parallel_config=NS(
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
            data_parallel_size=1,
            enable_expert_parallel=True,
        ),
    )
    group = NS(
        world_size=2,
        rank_in_group=rank,
        is_first_rank=rank == 0,
        is_last_rank=rank == 1,
    )
    return config, group


class OwnershipTests(unittest.TestCase):
    def test_no_ep_requires_exact_opt_in_and_preserves_ownership(self):
        for rank in (0, 1):
            config, group = fixture(rank)
            config.parallel_config.enable_expert_parallel = False
            env = {
                FLAG: "1",
                "VLLM_PP_LAYER_PARTITION": "25,23",
                "VLLM_FLASH_NO_EP_AB": "1",
            }
            self.assertEqual(should_omit(config, group, "embedding", env), rank == 1)
            self.assertEqual(should_omit(config, group, "lm_head", env), rank == 0)
            config.parallel_config.enable_expert_parallel = True
            with self.assertRaises(ValueError):
                should_omit(config, group, "lm_head", env)
            env["VLLM_FLASH_NO_EP_AB"] = "yes"
            with self.assertRaises(ValueError):
                should_omit(config, group, "lm_head", env)

    def test_default_off_does_not_inspect_configuration(self):
        for role in ("embedding", "lm_head"):
            self.assertFalse(should_omit(None, None, role, {}))
            self.assertFalse(should_omit(None, None, role, {FLAG: "0"}))

    def test_exact_roles_and_lazy_construction(self):
        with patch.dict(os.environ, {FLAG: "1", "VLLM_PP_LAYER_PARTITION": "25,23"}):
            for rank in (0, 1):
                config, group = fixture(rank)
                for role in ("embedding", "lm_head"):
                    calls = []
                    original = NS(named_parameters=lambda: [])
                    placeholder = NS(named_parameters=lambda: [])

                    def factory(calls=calls, original=original):
                        calls.append("allocate")
                        return original

                    def missing(name, calls=calls, placeholder=placeholder):
                        calls.append(name)
                        return placeholder

                    omit = (rank == 1) if role == "embedding" else (rank == 0)
                    result = build_pp_vocab_module(
                        config, group, role, factory, missing
                    )
                    self.assertIs(result, placeholder if omit else original)
                    self.assertEqual(
                        calls, ["unused_pp_" + role] if omit else ["allocate"]
                    )

    def test_invalid_feature_and_shape_gates(self):
        variants = [
            ("speculative_config", NS(method="mtp")),
            ("speculative_config", NS(method="dflash")),
            ("lora_config", NS()),
            ("model_config.hf_config.model_type", "other"),
            ("model_config.hf_config.tie_word_embeddings", True),
            ("model_config.hf_text_config.tie_word_embeddings", True),
            ("model_config.hf_text_config.num_hidden_layers", 47),
            ("model_config.hf_text_config.hidden_size", 4096),
            ("model_config.hf_text_config.vocab_size", 248321),
            ("model_config.multimodal_config.language_model_only", False),
            ("model_config.multimodal_config", None),
            ("parallel_config.pipeline_parallel_size", 1),
            ("parallel_config.tensor_parallel_size", 4),
            ("parallel_config.data_parallel_size", 2),
            ("parallel_config.enable_expert_parallel", False),
        ]
        config, group = fixture()
        for path, value in variants:
            candidate = copy.deepcopy(config)
            obj = candidate
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                should_omit(
                    candidate,
                    group,
                    "lm_head",
                    {FLAG: "1", "VLLM_PP_LAYER_PARTITION": "25,23"},
                )

    def test_invalid_group_environment_and_role(self):
        config, group = fixture()
        for key, value in [
            ("world_size", 3),
            ("rank_in_group", True),
            ("rank_in_group", 2),
            ("is_first_rank", False),
            ("is_last_rank", True),
        ]:
            candidate = copy.copy(group)
            setattr(candidate, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                should_omit(
                    config,
                    candidate,
                    "lm_head",
                    {FLAG: "1", "VLLM_PP_LAYER_PARTITION": "25,23"},
                )
        for env in (
            {FLAG: "yes"},
            {FLAG: "1"},
            {FLAG: "1", "VLLM_PP_LAYER_PARTITION": "26,22"},
        ):
            with self.assertRaises(ValueError):
                should_omit(config, group, "lm_head", env)
        with self.assertRaises(ValueError):
            should_omit(config, group, "unknown", {})


if __name__ == "__main__":
    unittest.main()
