import logging
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
)
from dify_plugin.interfaces.model.openai_compatible.llm import (
    OAICompatLargeLanguageModel,
)

logger = logging.getLogger(__name__)


class CerebrasLargeLanguageModel(OAICompatLargeLanguageModel):
    _REASONING_MODELS: set[str] = {
        "gpt-oss-120b",
        "zai-glm-4.6",
    }

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._add_custom_parameters(credentials)

        if stream and self._should_disable_stream(model, model_parameters, tools):
            logger.debug(
                "Disabling streaming for model %s due to Cerebras JSON/tool calling restrictions.",
                model,
            )
            stream = False

        return super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        super().validate_credentials(model, credentials)

    @staticmethod
    def _add_custom_parameters(credentials: dict) -> None:
        credentials["mode"] = "chat"

    def _should_disable_stream(
        self,
        model: str,
        model_parameters: Optional[dict],
        tools: Optional[list[PromptMessageTool]],
    ) -> bool:
        if model not in self._REASONING_MODELS:
            return False

        uses_json_mode = self._is_json_mode(model_parameters)
        uses_tool_calling = bool(tools)
        return uses_json_mode or uses_tool_calling

    @staticmethod
    def _is_json_mode(model_parameters: Optional[dict]) -> bool:
        if not model_parameters:
            return False

        response_format = model_parameters.get("response_format")
        if isinstance(response_format, str):
            if response_format.lower() in {"json", "json_object", "json_schema"}:
                return True
        elif isinstance(response_format, dict):
            response_type = response_format.get("type")
            if isinstance(response_type, str) and response_type.lower() in {"json", "json_object", "json_schema"}:
                return True

        if model_parameters.get("json_schema"):
            return True

        return False
