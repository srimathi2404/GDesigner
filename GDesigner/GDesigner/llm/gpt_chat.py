from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

# Standard OpenAI-compatible client - works against the real OpenAI API and
# against any server that mirrors its /v1/chat/completions contract (e.g.
# `vllm serve ... --served-model-name <alias>`).
aclient = AsyncOpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEYS)


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],
    max_tokens: int = 1000,
    temperature: float = 0.2,
    ):
    response = await aclient.chat.completions.create(
        model=model,
        messages=msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    prompt = "".join([item['content'] for item in msg])
    cost_count(prompt, answer, model)
    return answer

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return await achat(self.model_name, messages, max_tokens=max_tokens, temperature=temperature)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass