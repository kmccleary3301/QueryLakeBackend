from ray.serve.handle import DeploymentResponseGenerator
from typing import Awaitable, Callable, AsyncGenerator, List
import json, inspect

async def stream_results_tokens(results_generator: DeploymentResponseGenerator,
                                encode_output : bool = False,
                                on_new_token: Awaitable[Callable[[str], None]] = None,
                                stop_sequences: List[str] = None) -> AsyncGenerator[bytes, None]:
    
    num_returned, tokens_returned, stop_queue, hold_queue = 0, [], [], False
    
    async def new_token_call(text_input):
        nonlocal tokens_returned
        if not on_new_token is None:
            if inspect.iscoroutinefunction(on_new_token):
                await on_new_token(text_input)
            else:
                on_new_token(text_input)
        
        tokens_returned.append(text_input)
    
    def check_stop_sequence(text_in):
        if text_in == "":
            return False
        if stop_sequences is not None:
            for stop_sequence in stop_sequences:
                if stop_sequence.startswith(text_in):
                    return True
        return False
    
    def yield_function(text_in):
        return (json.dumps({"text": text_in}) + "\n").encode("utf-8") if encode_output else text_in
    
    async for request_output in results_generator:
        text_outputs = [output.text for output in request_output.outputs]
        assert len(text_outputs) == 1
        text_output = text_outputs[0][num_returned:]
        num_returned = num_returned + len(text_output)
        
        # The following code is responsible for withholding the output if 
        # a stop sequence is being matched. This avoids a partial stop sequence
        # being returned just before termination.
        if stop_sequences is not None:
            if not hold_queue:
                for i in range(len(text_output)):
                    match_stop = check_stop_sequence(text_output[i:])
                    if match_stop:
                        last_valid = text_output[:i]
                        if len(last_valid) > 0:
                            await new_token_call(last_valid)
                            yield yield_function(last_valid)
                        stop_queue.append(text_output[i:])
                        hold_queue = True
                        break
            else:
                stop_queue.append(text_output)
                stop_queue_full = "".join(stop_queue)
                if (any([stop_sequence in stop_queue_full for stop_sequence in stop_sequences])):
                    print("Stopping sequence found:", stop_queue_full)
                    return
                elif (check_stop_sequence(stop_queue_full)):
                    continue
                else:
                    text_output = stop_queue_full
                    hold_queue = False
                    stop_queue = []
        
        
        if not hold_queue:
            await new_token_call(text_output)
            yield yield_function(text_output)
            
            
async def consume_deployment_response(results_generator: DeploymentResponseGenerator) -> List[str]:
    token_list, i = [], 0
    async for result in results_generator:
        token_list.append(result.output)
    if len(token_list) == 1:
        token_list = token_list[0].outputs
    return token_list