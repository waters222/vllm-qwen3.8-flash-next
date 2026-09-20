"""Default-off, one-deferral admission pilot. Never changes running cadence.

Completing chunks of already-running prefills are conservatively ambiguous and
bypass balancing. Only fresh same-loop prefills and established decodes count.
This deliberately does not implement running/chunked-prefill phase correction.
"""
import os


class PhaseBalance:
    @classmethod
    def from_env(cls, scheduler, environ=None):
        env=os.environ if environ is None else environ
        flag=env.get('VLLM_FLASH_PP_BALANCE','0')
        if flag not in ('0','1'):
            raise ValueError('VLLM_FLASH_PP_BALANCE must be 0 or 1')
        if flag=='0':return None
        s=scheduler
        if not (
            type(s).__module__=='vllm.v1.core.sched.async_scheduler'
            and type(s).__name__=='AsyncScheduler'
            and s.use_v2_model_runner and s.scheduler_config.async_scheduling
            and s.parallel_config.pipeline_parallel_size==2
            and s.parallel_config.data_parallel_size==1
            and s.vllm_config.model_config.hf_config.model_type=='qwen4_exp'
            and s.vllm_config.speculative_config is None and s.num_spec_tokens==0
            and s.num_sampled_tokens_per_step==1
            and s.lora_config is None and not s.cache_config.enable_prefix_caching
            and s.connector is None and s.ec_connector is None
            and not s.is_encoder_decoder and not s.is_mm_encoder_only
            and s.max_num_encoder_input_tokens==0
        ):
            raise ValueError('Phase balance requires standard Qwen4Exp V2 async PP2 DP1 text-only serving without speculation, LoRA, prefix cache or connectors')
        return cls()

    def __init__(self):
        # Internal identifiers only; values are numeric identities/steps, never references.
        self._deferred={}

    def clear(self, request):
        marker=self._deferred.get(request.request_id)
        if marker is not None and marker[0]==id(request):
            del self._deferred[request.request_id]

    def _prune(self, scheduler):
        self._deferred={key:value for key,value in self._deferred.items()
                        if key in scheduler.requests and id(scheduler.requests[key])==value[0]}

    @staticmethod
    def _status(request, name):
        return getattr(request.status,'name',None)==name

    @staticmethod
    def _plain(request):
        return (
            request.num_preemptions==0 and request.num_stale_output_tokens==0
            and not request.resumable and not request.use_structured_output
            and getattr(request,'session_id',None) is None
            and not request.has_encoder_inputs and not request.mm_features
            and request.lora_request is None and request.prompt_token_ids is not None
            and getattr(request,'prompt_embeds',None) is None
            and getattr(request,'inputs_embeds',None) is None
            and not request.spec_token_ids
        )

    @classmethod
    def _fresh(cls, request):
        return (cls._plain(request) and request.num_computed_tokens==0
                and request.num_output_tokens==0 and request.num_output_placeholders==0
                and request.next_decode_eligible_step==0
                and 1<=request.num_prompt_tokens<=512
                and request.num_tokens==request.num_prompt_tokens)

    @staticmethod
    def _completes(request, computed, scheduled):
        # Exact base Scheduler._update_after_schedule prefill predicate, negated.
        return computed+scheduled >= request.num_tokens+request.num_output_placeholders

    def should_defer(self, scheduler, request, num_new_tokens, num_computed_tokens,
                     num_scheduled_tokens, load_kv_async, num_external_computed_tokens):
        self._prune(scheduler)
        if (load_kv_async or num_external_computed_tokens or num_computed_tokens!=0
                or scheduler.num_waiting_for_streaming_input
                or not self._status(request,'WAITING') or not self._fresh(request)
                or num_new_tokens!=request.num_prompt_tokens
                or not self._completes(request,num_computed_tokens,num_new_tokens)
                or scheduler.requests.get(request.request_id) is not request):
            return False
        if request.request_id in self._deferred:
            return False # Never repeatedly postpone the same waiting request.
        step=scheduler.current_step
        occupancy=[0,0]
        seen=set()
        for running in scheduler.running:
            if running is request or id(running) in seen:
                return False # Unexpected duplicate running identity: do not guess.
            seen.add(id(running))
            if not self._status(running,'RUNNING') or not self._plain(running):return False
            n=num_scheduled_tokens.get(running.request_id,0)
            computed=running.num_computed_tokens
            if n:
                if n<1 or not self._completes(running,computed,n):return False
                if self._fresh(running):
                    if n!=running.num_prompt_tokens:return False
                    # Includes first successful admission in the current waiting loop.
                elif (n!=1 or running.is_prefill_chunk or computed<running.num_prompt_tokens
                      or not 0<=running.num_output_placeholders<=1):
                    return False
                occupancy[(step+2)%2]+=1
            else:
                if (running.is_prefill_chunk or computed<running.num_prompt_tokens
                        or not 0<=running.num_output_placeholders<=1):return False
                eligible=running.next_decode_eligible_step
                if not step<eligible<=step+2:return False # Overdue/unscheduled is ambiguous.
                occupancy[eligible%2]+=1
        phase=(step+2)%2
        if occupancy[phase]>occupancy[1-phase]:
            self._deferred[request.request_id]=(id(request),step)
            return True
        return False
