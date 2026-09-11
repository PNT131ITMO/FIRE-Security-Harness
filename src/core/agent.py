from src.utils import shared_config, text_utils

SYS_PROMT = 'You are a fact-checking agent responsible for verifying the accuracy of claims.'
AGNES_API_BASE = 'https://apihub.agnes-ai.com/v1'
GROQ_API_BASE = 'https://api.groq.com/openai/v1'

class Model:
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.5,
            max_tokens: int = 2048,
            show_responses: bool = False,
            show_prompts: bool = False,
            request_timeout: float = 60,
            max_api_retries: int = 2,
    ) -> None:
        if not isinstance(model_name, str) or ':' not in model_name:
            raise ValueError('model_name must use provider:model_id format.')

        self.organization, self.model_id = model_name.split(':', 1)

        if not self.organization or not self.model_id.strip():
            raise ValueError('model_name must use provider:model_id format.')
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
            raise ValueError('temperature must be between 0 and 2.')
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer.')
        if not isinstance(request_timeout, (int, float)) or isinstance(request_timeout, bool) or not 0 < request_timeout < float('inf'):
            raise ValueError('request_timeout must be finite and positive.')
        if type(max_api_retries) is not int or max_api_retries < 0:
            raise ValueError('max_api_retries must be a non-negative integer.')

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_responses = show_responses
        self.show_prompts = show_prompts
        self.request_timeout = request_timeout
        self.max_api_retries = max_api_retries
        self.model = self.load_model()

    def load_model(self):
        loaders = {
            'openai': self._load_openai_model,
            'groq': self._load_groq_model,
            'anthropic': self._load_anthropic_model,
            'hf': self._load_huggingface_model,
            'agnes': self._load_agnes_model,
        }

        if self.organization not in loaders:
            raise ValueError(f'Unsupported organization: {self.organization}')

        return loaders[self.organization]
    
    def _load_openai_model(self):
        api_key = shared_config.get_api_key('OPENAI_API_KEY')
        from langchain_openai import ChatOpenAI

        options = {
            'model': self.model_id,
            'openai_api_key': api_key,
            'temperature': self.temperature,
            'timeout': self.request_timeout,
            'max_retries': self.max_api_retries,
        }

        if self.model_id.startswith('o1'):
            options['temperature'] = 1
            options['model_kwargs'] = {'max_completion_tokens': self.max_tokens}
        else:
            options['max_tokens'] = self.max_tokens
        return ChatOpenAI(**options)
    
    def _load_anthropic_model(self):
        api_key = shared_config.get_api_key('ANTHROPIC_API_KEY')
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model_id,
            anthropic_api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
            max_retries=self.max_api_retries,
        )

    def _load_huggingface_model(self):
        import torch
        import transformers

        cuda = torch.cuda.is_available()
        dtype = torch.float32

        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return transformers.pipeline(
            'text-generation',
            model=self.model_id,
            model_kwargs={'torch_dtype': dtype},
            device=0 if cuda else -1,
            max_new_tokens=self.max_tokens
        )
    
    def _load_groq_model(self):
        return self._load_compatible_model('GROQ_API_KEY', GROQ_API_KEY)
    
    def _load_agnes_model(self):
        return self._load_compatible_model('AGNES_API_KEY', AGNES_API_KEY)
    
    def _load_compatible_model(self, env_var_name: str, base_url: str):
        api_key = shared_config.get_api_key(env_var_name)

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=api_key,
            openai_api_base=base_url,
            timeout=self.request_timeout,
            max_retries=self.max_api_retries,
        )
    
    def generate(self, context: str) -> tuple[str, dict | None]:
        if self.show_prompts:
            print(context)
        if self.organization in {'openai', 'agnes', 'groq'}:
            result = self._generate_openai_response(context)
        elif self.organization == 'anthropic':
            result = self._generate_anthropic_response(context)
        elif self.organization == 'hf':
            result = self._generate_huggingface_response(context)
        else:
            raise ValueError(f'Unsupported organization: {self.organization}')
        

        if self.show_responses:
            print(result[0])
        
        return result

    @staticmethod
    def _read_response(response) -> tuple[str, dict | None]:
        content = response.content
        if isinstance(content, list):
            content = '\n'.join(
                block if isinstance(block, str) else block['text']
                for block in content
                if isinstance(block, str) or (
                    isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))
            )
        if not isinstance(content, str):
            content = ''
        
        usage = getattr(response, 'response_metadata', {}) or {}

        if usage is None:
            metadata = getattr(response, 'response_metadata', {}) or {}
            usage = metadata.get('usage') or metadata.get('token_usage')

            if usage and 'prompt_tokens' in usage:
                usage = {
                    'input_tokens': usage.get('prompt_tokens'),
                    'output_tokens': usage.get('completion_tokens'),
                }
        return content, usage

    def _generate_openai_response(self, context: str) -> tuple[str, dict | None]:
        if self.organization == 'openai' and self.model_id.startswith('o1'):
            response = self.model.invoke(context)
        else:
            response = self.model.invoke([('system', SYS_PROMT), ('human', context)])
        return self._read_response(response)
    
    def _generate_anthropic_response(self, context: str) -> tuple[str, dict | None]:
        response = self.model.invoke([('system', SYS_PROMT), ('human', context)])
        return self._read_response(response)
    
    def _generate_huggingface_response(self, context: str) -> tuple[str, dict | None]:
        messages = [
            {'role': 'system', 'content': SYS_PROMT},
            {'role': 'user', 'content': context},
        ]

        options = {
            'pad_token_id': self.model.tokenizer.eos_token_id,
            'do_sample': self.temperature > 0,
        }

        if self.temperature > 0:
            options['temperature'] = self.temperature

        outputs = self.model(messages, **options)

        return outputs[0]['generated_text'][-1]['content'], None
    
    def print_config(self) -> None:
        settings = {
            'organization': self.organization,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'show_responses': self.show_responses,
            'show_prompts': self.show_prompts,
            'request_timeout': self.request_timeout,
            'max_api_retries': self.max_api_retries,
        }
        print(text_utils.to_readable_json(settings))