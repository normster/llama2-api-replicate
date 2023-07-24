# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from copy import deepcopy
import json
from vllm import LLM, SamplingParams, CompletionOutput, RequestOutput, sequence


def serialize_completion(output: CompletionOutput):
    return {
        "index": output.index,
        "text": output.text,
        "token_ids": output.token_ids,
        "cumulative_logprob": output.cumulative_logprob,
        "logprobs": output.logprobs,
        "finish_reason": output.finish_reason,
    }


def serialize_response(outputs: RequestOutput):
    return {
        "request_id": outputs.request_id,
        "prompt": outputs.prompt,
        "prompt_token_ids": outputs.prompt_token_ids,
        "outputs": [serialize_completion(o) for o in outputs.outputs],
        "finished": outputs.finished,
    }


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.default_params = {
            "temperature": 0.,
            "max_tokens": 100,
        }
        self.model = LLM(model="./weights")

    def _reset_requests(self, scheduler):
        """Manually remove all existing requests in LLMEngine's scheduler."""
        for state_queue in [scheduler.waiting, scheduler.running, scheduler.swapped]:
            for seq_group in state_queue:
                # Remove the sequence group from the state queue.
                state_queue.remove(seq_group)
                for seq in seq_group.seqs:
                    if seq.is_finished():
                        continue
                    scheduler.free_seq(seq, sequence.SequenceStatus.FINISHED_ABORTED)

    def predict(
        self,
        inputs: str = Input(
            description="JSON-encoded input data with keys 'prompts' and 'params'"
        ),
    ) -> str:
        self._reset_requests(self.model.llm_engine.scheduler)
        inputs = json.loads(inputs)
        params = deepcopy(self.default_params)
        params.update(inputs.get("params", {}))

        params = SamplingParams(**params)
        prompts = inputs["prompts"]

        if not isinstance(prompts, list):
            prompts = [prompts]

        status = ""
        generations = []
        try:
            generations = self.model.generate(prompts, params)
            generations = [serialize_response(g) for g in generations]
        except Exception as e:
            status = "Error: " + str(e)
            pass

        response = {"generations": generations, "status": status}
        return json.dumps(response)
