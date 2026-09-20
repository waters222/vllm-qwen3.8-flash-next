"""Default-off PP vocabulary ownership for the reviewed text-only trial.

Do not reuse this gate for multimodal, tied-weight, LoRA or speculative models.
Only unused modules are omitted; retained factories and checkpoint values are
unchanged. Missing modules must be loader-recognized, fail-fast placeholders.
"""
import logging
import os


FLAG = 'VLLM_FLASH_PP_VOCAB_OWNERS_ONLY'


def should_omit(config, group, role, environ=None):
    if role not in ('embedding', 'lm_head'):
        raise ValueError('Unknown PP vocabulary role')
    env = os.environ if environ is None else environ
    flag = env.get(FLAG, '0')
    if flag not in ('0', '1'):
        raise ValueError(f'{FLAG} must be 0 or 1')
    if flag == '0':
        return False
    model = config.model_config
    parallel = config.parallel_config
    text = model.hf_text_config
    multimodal = model.multimodal_config
    no_ep = env.get('VLLM_FLASH_NO_EP_AB', '0')
    if no_ep not in ('0', '1'):
        raise ValueError('VLLM_FLASH_NO_EP_AB must be 0 or 1')
    if not (
        model.hf_config.model_type == 'qwen4_exp'
        and multimodal is not None and multimodal.language_model_only is True
        and model.hf_config.tie_word_embeddings is False
        and text.tie_word_embeddings is False
        and text.num_hidden_layers == 48 and text.hidden_size == 2560
        and text.vocab_size == 248320
        and config.speculative_config is None and config.lora_config is None
        and parallel.pipeline_parallel_size == 2
        and parallel.tensor_parallel_size == 2 and parallel.data_parallel_size == 1
        and parallel.enable_expert_parallel is (no_ep == '0')
        and group.world_size == 2 and type(group.rank_in_group) is int
        and group.rank_in_group in (0, 1)
        and group.is_first_rank == (group.rank_in_group == 0)
        and group.is_last_rank == (group.rank_in_group == 1)
        and env.get('VLLM_PP_LAYER_PARTITION') == '25,23'
    ):
        raise ValueError('PP vocab ownership requires the reviewed untied text-only '
                         'Qwen4Exp TP2/PP2 25/23 configuration with matched EP A/B opt-in, without LoRA or speculation')
    return not (group.is_first_rank if role == 'embedding' else group.is_last_rank)


def build_pp_vocab_module(config, group, role, factory, missing_factory):
    omit = should_omit(config, group, role)
    if os.environ.get(FLAG, '0') == '0':
        return factory()
    module = missing_factory('unused_pp_' + role) if omit else factory()
    inventory = [(name, tuple(weight.shape), str(weight.dtype), weight.device.type)
                 for name, weight in module.named_parameters()]
    if omit and inventory:
        raise RuntimeError('Omitted PP vocabulary module unexpectedly owns parameters')
    logging.getLogger(__name__).info(
        'PP vocabulary ownership: role=%s pp_rank=%s omitted=%s parameter_inventory=%s',
        role, group.rank_in_group, omit, inventory)
    return module
