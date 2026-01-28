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
    Um wrapper (adaptador) em torno do LangChain BaseChatModel que implementa
    o protocolo BaseChatModel do browser-use.
    """
    chat: 'LangChainBaseChatModel'

    @property
    def model(self) -> str:
        return self.name

    @property
    def provider(self) -> str:
        """Retorna o nome do provedor baseado na classe do modelo LangChain."""
        model_class_name = self.chat.__class__.__name__.lower()
        if 'openai' in model_class_name:
            return 'openai'
        elif 'ollama' in model_class_name:
            return 'ollama'
        return 'langchain'

    @property
    def name(self) -> str:
        """Retorna o nome do modelo de forma segura."""
        model_name = getattr(self.chat, 'model_name', None)
        if model_name:
            return str(model_name)
        model_attr = getattr(self.chat, 'model', None)
        if model_attr:
            return str(model_attr)
        return self.chat.__class__.__name__

    def _get_usage(self, response: 'LangChainAIMessage') -> ChatInvokeUsage | None:
        """Extrai metadados de uso de tokens."""
        if not hasattr(response, 'usage_metadata') or response.usage_metadata is None:
            return None
            
        usage = response.usage_metadata
        return ChatInvokeUsage(
            prompt_tokens=usage.get('input_tokens', 0),
            completion_tokens=usage.get('output_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
        )

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Invoca o modelo LangChain com as mensagens fornecidas.
        """
        langchain_messages = LangChainMessageSerializer.serialize_messages(messages)

        try:
            if output_format is None:
                # Retorna resposta string simples
                response = await self.chat.ainvoke(langchain_messages)
                content = response.content if hasattr(response, 'content') else str(response)
                return ChatInvokeCompletion(
                    completion=str(content),
                    usage=self._get_usage(response),
                )

            else:
                # Tenta usar structured output
                try:
                    # Método preferido: suporte nativo a JSON
                    structured_chat = self.chat.with_structured_output(output_format)
                    parsed_object = await structured_chat.ainvoke(langchain_messages)
                    return ChatInvokeCompletion(completion=parsed_object, usage=None)
                except Exception:
                    response = await self.chat.ainvoke(langchain_messages)
                    content = str(response.content)

                    # Limpeza de Markdown se necessário (ex: ```json ... ```)
                    if "```json" in content:
                        content = content.replace("```json", "").replace("```", "").strip()
                    elif "```" in content:
                        content = content.replace("```", "").strip()

                    import json
                    parsed_data = json.loads(content)
                    
                    # Valida e converte para o modelo Pydantic esperado
                    parsed_object = output_format(**parsed_data)
                    
                    return ChatInvokeCompletion(
                        completion=parsed_object,
                        usage=self._get_usage(response),
                    )

        except Exception as e:
            raise ModelProviderError(
                message=f'LangChain wrapper error: {str(e)}',
                model=self.name,
            ) from e