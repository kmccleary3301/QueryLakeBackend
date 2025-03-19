from QueryLake.typing.config import AuthType
from typing import (
    Union, 
    List, 
    Callable, 
    Dict, 
    Awaitable, 
    Tuple,
    Literal
)
from asyncio import gather
from ..api import api

async def embedding_call(
    self, # Umbrella class, can't type hint because of circular imports
    auth : AuthType,
    inputs : List[str],
    model: str = None,
    return_tokens_usage = False
):
    assert self.config.enabled_model_classes.embedding, "Embedding models are disabled on this QueryLake Deployment"
    if model is None:
        model = self.config.default_models.embedding
    assert model in self.embedding_handles, f"Model choice [{model}] not available for embeddings"
    (user, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
    result = await self.embedding_handles[model].run.remote({"text": inputs})
    
    if isinstance(result, list):
        embedding = [e["embedding"] for e in result]
        total_tokens = sum([e["token_count"] for e in result])
    else:
        embedding = result["embedding"]
        total_tokens = result["token_count"]
    
    api.increment_usage_tally(self.database, user_auth, {
        "embedding": {
            self.config.default_models.embedding: {"tokens": total_tokens}
        }
    }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
    
    return embedding if not return_tokens_usage else (embedding, total_tokens)

async def rerank_call(
    self, # Umbrella class, can't type hint because of circular imports
    auth : AuthType,
    inputs : Union[List[Tuple[str, str]], Tuple[str, str]],
    normalize : Union[bool, List[bool]] = True,
    model: str = None
):
    assert self.config.enabled_model_classes.rerank, "Rerank models are disabled on this QueryLake Deployment"
    if model is None:
        model = self.config.default_models.rerank
    assert model in self.rerank_handles, f"Model choice [{model}] not available for rerankers"
    (user, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
    
    if isinstance(inputs, list):
        if not isinstance(normalize, list):
            normalize = [normalize for _ in range(len(inputs))]
        assert len(normalize) == len(inputs), \
            "All input lists must be the same length"
        result = await gather(*[self.rerank_handles[model].run.remote(
            inputs[i],
            normalize=normalize[i]
        ) for i in range(len(inputs))])
        scores = [e["score"] for e in result]
        total_tokens = sum([e["token_count"] for e in result])
        
    else:
        result = await self.rerank_handles[model].run.remote(inputs, normalize=normalize)
        scores = result["score"]
        total_tokens = result["token_count"]
    
    api.increment_usage_tally(self.database, user_auth, {
        "rerank": {
            self.config.default_models.rerank: {"tokens": total_tokens}
        }
    }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
    
    return scores

async def web_scrape_call(
    self, # Umbrella class, can't type hint because of circular imports
    auth : AuthType,
    inputs : Union[str, List[str]],
    timeout : Union[float, List[float]] = 10,
    markdown : Union[bool, List[bool]] = True,
    load_strategy : Union[Literal["full", "eager", "none"], list] = "full",
    summary: Union[bool, List[bool]] = False
):
    
    (_, _) = api.get_user(self.database, auth)
    
    if isinstance(inputs, list):
        if not isinstance(timeout, list):
            timeout = [timeout for _ in range(len(inputs))]
        if not isinstance(markdown, list):
            markdown = [markdown for _ in range(len(inputs))]
        if not isinstance(summary, list):
            summary = [summary for _ in range(len(inputs))]
        if not isinstance(load_strategy, list):
            load_strategy = [load_strategy for _ in range(len(inputs))]
        
        assert all([len(timeout) == len(inputs), len(markdown) == len(inputs), len(summary) == len(inputs)]), \
            "All input lists must be the same length"   
        
        return await gather(*[self.web_scraper_handle.run.remote(
            inputs[i],
            timeout=timeout[i],
            markdown=markdown[i],
            summary=summary[i]
        ) for i in range(len(inputs))])
    else:
        return await self.web_scraper_handle.run.remote(inputs, timeout, markdown, summary)