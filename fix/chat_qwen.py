from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Any

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

from serializer import LangChainMessageSerializer

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel
    from langchain_core.messages import AIMessage as LangChainAIMessage

T = TypeVar('T', bound=BaseModel)

@dataclass
class ChatLangchain(BaseChatModel):
    """
    Wrapper oficial do Browser-Use para modelos LangChain.
    Versão com correção de JSON (Qwen) e Metadados de Uso (Browser-Use 0.1.4+).
    """
    chat: 'LangChainBaseChatModel'

    @property
    def model(self) -> str:
        return self.name

    @property
    def provider(self) -> str:
        model_class_name = self.chat.__class__.__name__.lower()
        if 'openai' in model_class_name:
            return 'openai'
        elif 'ollama' in model_class_name:
            return 'ollama'
        return 'langchain'

    @property
    def name(self) -> str:
        model_name = getattr(self.chat, 'model_name', None)
        if model_name:
            return str(model_name)
        model_attr = getattr(self.chat, 'model', None)
        if model_attr:
            return str(model_attr)
        return self.chat.__class__.__name__

    def _get_usage(self, response: 'LangChainAIMessage') -> ChatInvokeUsage | None:
        """
        Extrai metadados de uso e preenche campos obrigatórios para evitar erros de validação.
        """
        if not hasattr(response, 'usage_metadata') or response.usage_metadata is None:
            # Retorna um objeto zerado em vez de None, para segurança
            return ChatInvokeUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_cached_tokens=0,
                prompt_cache_creation_tokens=0,
                prompt_image_tokens=0
            )
            
        usage = response.usage_metadata
        
        # AQUI ESTÁ A CORREÇÃO DO SEU ERRO ATUAL:
        # Adicionamos os campos *_tokens com valor 0, pois o llama-swap não os envia.
        return ChatInvokeUsage(
            prompt_tokens=usage.get('input_tokens', 0),
            completion_tokens=usage.get('output_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
            prompt_cached_tokens=usage.get('prompt_cached_tokens', 0),
            prompt_cache_creation_tokens=usage.get('prompt_cache_creation_tokens', 0),
            prompt_image_tokens=usage.get('prompt_image_tokens', 0),
        )

    def _fix_json_issues(self, data: Any) -> Any:
        """
        Corrige APENAS os parâmetros internos (element->index, value->text).
        """
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                processed_v = self._fix_json_issues(v)
                
                if isinstance(processed_v, dict):
                    if 'element' in processed_v and 'index' not in processed_v:
                        processed_v['index'] = processed_v.pop('element')
                    
                    if 'value' in processed_v and 'text' not in processed_v:
                        processed_v['text'] = processed_v.pop('value')

                new_data[k] = processed_v
            
            if 'element' in new_data and 'index' not in new_data:
                new_data['index'] = new_data.pop('element')
                
            return new_data
            
        elif isinstance(data, list):
            return [self._fix_json_issues(item) for item in data]
            
        return data

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        
        langchain_messages = LangChainMessageSerializer.serialize_messages(messages)

        try:
            response = await self.chat.ainvoke(langchain_messages)
            content = str(response.content)

            if "```json" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            elif "```" in content:
                content = content.replace("```", "").strip()

            if output_format:
                import json
                try:
                    parsed_data = json.loads(content)
                    
                    # Aplica a correção de JSON (Isso já estava funcionando)
                    fixed_data = self._fix_json_issues(parsed_data)
                    
                    parsed_object = output_format(**fixed_data)
                    
                    return ChatInvokeCompletion(
                        completion=parsed_object,
                        # Agora este método não vai mais falhar
                        usage=self._get_usage(response),
                    )
                except json.JSONDecodeError:
                    raise ModelProviderError(f"Model output not valid JSON: {content}", model=self.name)
                except Exception as e:
                    raise ModelProviderError(f"Validation Error. Raw: {content} | Fixed: {fixed_data} | Error: {e}", model=self.name)
            
            return ChatInvokeCompletion(
                completion=content,
                usage=self._get_usage(response),
            )

        except Exception as e:
            raise ModelProviderError(
                message=f'LangChain wrapper error: {str(e)}',
                model=self.name,
            ) from e