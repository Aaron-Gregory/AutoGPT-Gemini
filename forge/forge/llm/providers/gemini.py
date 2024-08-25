from __future__ import annotations

import asyncio
import enum
import logging
from typing import Any, Callable, Optional, ParamSpec, Sequence, TypeVar

import sentry_sdk
import tenacity
import tiktoken

# TODO: These are the wrong type of errors, we're just not retries with Gemini til they're fixed
from anthropic import APIConnectionError, APIStatusError
from pydantic import SecretStr

from forge.models.config import UserConfigurable

from .schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    BaseChatModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    ToolResultMessage,
)
from .utils import validate_tool_calls

_T = TypeVar("_T")
_P = ParamSpec("_P")


class GeminiModelName(str, enum.Enum):
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_0_PRO = "gemini-1.0-pro"


GEMINI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=GeminiModelName.GEMINI_1_5_FLASH,
            provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=0.075 / 1e6,
            completion_token_cost=0.3 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GeminiModelName.GEMINI_1_5_PRO,
            provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=3.5 / 1e6,
            completion_token_cost=10.5 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GeminiModelName.GEMINI_1_0_PRO,
            provider_name=ModelProviderName.GEMINI,
            prompt_token_cost=0.5 / 1e6,
            completion_token_cost=1.5 / 1e6,
            max_tokens=200000,
            has_function_call_api=True,
        ),
    ]
}


class GeminiCredentials(ModelProviderCredentials):
    """Credentials for Gemini."""

    api_key: SecretStr = UserConfigurable(from_env="GEMINI_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="GEMINI_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class GeminiSettings(ModelProviderSettings):
    credentials: Optional[GeminiCredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore


class GeminiProvider(BaseChatModelProvider[GeminiModelName, GeminiSettings]):
    default_settings = GeminiSettings(
        name="gemini_provider",
        description="Provides access to Gemini's API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: GeminiSettings
    _credentials: GeminiCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[GeminiSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.model_copy(deep=True)
        if not settings.credentials:
            settings.credentials = GeminiCredentials.from_env()

        super(GeminiProvider, self).__init__(settings=settings, logger=logger)

        import google.generativeai as gemini

        gemini.configure(**self._credentials.get_api_access_kwargs())

        self._client = gemini

    async def get_available_models(self) -> Sequence[ChatModelInfo[GeminiModelName]]:
        return await self.get_available_chat_models()

    async def get_available_chat_models(
        self,
    ) -> Sequence[ChatModelInfo[GeminiModelName]]:
        return list(GEMINI_CHAT_MODELS.values())

    def get_token_limit(self, model_name: GeminiModelName) -> int:
        """Get the token limit for a given model."""
        return GEMINI_CHAT_MODELS[model_name].max_tokens

    def get_tokenizer(self, model_name: GeminiModelName) -> ModelTokenizer[Any]:
        # HACK: No official tokenizer is available for Claude 3
        return tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str, model_name: GeminiModelName) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: GeminiModelName,
    ) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: GeminiModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Gemini API."""
        system_instruction, gemini_messages, completion_kwargs = (
            self._get_chat_completion_args(
                prompt_messages=model_prompt,
                functions=functions,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
        )

        total_cost = 0.0
        attempts = 0
        while True:
            completion_kwargs["contents"] = gemini_messages.copy()
            if prefill_response:
                completion_kwargs["contents"].append(
                    {"role": "model", "parts": prefill_response}
                )

            (
                _assistant_msg,
                cost,
                t_input,
                t_output,
            ) = await self._create_chat_completion(
                system_instruction, model_name, completion_kwargs
            )
            total_cost += cost
            self._logger.debug(
                f"Completion usage: {t_input} input, {t_output} output "
                f"- ${round(cost, 5)}"
            )

            # Merge prefill into generated response
            if prefill_response:
                first_text_block = next(
                    b for b in _assistant_msg.candidates[0].content.parts if b.text
                )
                first_text_block.text = prefill_response + first_text_block.text

            assistant_msg = AssistantChatMessage(
                content="\n\n".join(
                    b.text for b in _assistant_msg.candidates[0].content.parts if b.text
                ),
                tool_calls=self._parse_assistant_tool_calls(_assistant_msg),
            )

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            tool_call_errors = []
            try:
                # Validate tool calls
                if assistant_msg.tool_calls and functions:
                    tool_call_errors = validate_tool_calls(
                        assistant_msg.tool_calls, functions
                    )
                    if tool_call_errors:
                        raise ValueError(
                            "Invalid tool use(s):\n"
                            + "\n".join(str(e) for e in tool_call_errors)
                        )

                parsed_result = completion_parser(assistant_msg)
                break
            except Exception as e:
                self._logger.debug(
                    f"Parsing failed on response: '''{_assistant_msg}'''"
                )
                self._logger.warning(f"Parsing attempt #{attempts} failed: {e}")
                sentry_sdk.capture_exception(
                    error=e,
                    extras={"assistant_msg": _assistant_msg, "i_attempt": attempts},
                )
                if attempts < self._configuration.fix_failed_parse_tries:
                    gemini_messages.append(
                        _assistant_msg.model_dump(include={"role", "parts"})  # type: ignore # noqa
                    )
                    gemini_messages.append(
                        {
                            "role": "user",
                            "parts": [
                                *(
                                    # tool_result is required if last assistant message
                                    # had tool_use block(s)
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tc.id,
                                        "is_error": True,
                                        "parts": [
                                            {
                                                "type": "text",
                                                "text": (
                                                    "Not executed because parsing "
                                                    "of your last message failed"
                                                    if not tool_call_errors
                                                    else (
                                                        str(e)
                                                        if (
                                                            e := next(
                                                                (
                                                                    tce
                                                                    for tce in tool_call_errors
                                                                    if tce.name
                                                                    == tc.function.name
                                                                ),
                                                                None,
                                                            )
                                                        )
                                                        else "Not executed because validation "
                                                        "of tool input failed"
                                                    )
                                                ),
                                            }
                                        ],
                                    }
                                    for tc in assistant_msg.tool_calls or []
                                ),
                                {
                                    "type": "text",
                                    "text": (
                                        "ERROR PARSING YOUR RESPONSE:\n\n"
                                        f"{e.__class__.__name__}: {e}"
                                    ),
                                },
                            ],
                        }
                    )
                else:
                    raise

        if attempts > 1:
            self._logger.debug(
                f"Total cost for {attempts} attempts: ${round(total_cost, 5)}"
            )

        return ChatModelResponse(
            response=assistant_msg,
            parsed_result=parsed_result,
            llm_info=GEMINI_CHAT_MODELS[model_name],
            prompt_tokens_used=t_input,
            completion_tokens_used=t_output,
        )

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Prepare arguments for message completion API call.

        Args:
            prompt_messages: List of ChatMessages.
            functions: Optional list of functions available to the LLM.
            kwargs: Additional keyword arguments.

        Returns:
            list[MessageParam]: Prompt messages for the Gemini call
            dict[str, Any]: Any other kwargs for the Gemini call
        """
        if functions:
            kwargs["tools"] = [
                {
                    "name": f.name,
                    "description": f.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            name: param.to_dict()
                            for name, param in f.parameters.items()
                        },
                        "required": [
                            name
                            for name, param in f.parameters.items()
                            if param.required
                        ],
                    },
                }
                for f in functions
            ]

        kwargs["max_tokens"] = max_output_tokens or 4096

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        system_messages = [
            m for m in prompt_messages if m.role == ChatMessage.Role.SYSTEM
        ]
        if (_n := len(system_messages)) > 1:
            self._logger.warning(
                f"Prompt has {_n} system messages; Gemini supports only 1 for now. "
                "They will be merged, and removed from the rest of the prompt."
            )
        system_instruction = "\n\n".join(sm.content for sm in system_messages)

        messages = []
        for message in prompt_messages:
            if message.role == ChatMessage.Role.SYSTEM:
                pass
            elif message.role == ChatMessage.Role.USER:
                # Merge subsequent user messages
                if messages and (prev_msg := messages[-1])["role"] == "user":
                    if isinstance(prev_msg["parts"], str):
                        prev_msg["parts"] += f"\n\n{message.content}"
                    else:
                        assert isinstance(prev_msg["parts"], list)
                        prev_msg["parts"].append(
                            {"type": "text", "text": message.content}
                        )
                else:
                    messages.append({"role": "user", "parts": message.content})
                # TODO: add support for image blocks
            elif message.role == ChatMessage.Role.ASSISTANT:
                if isinstance(message, AssistantChatMessage) and message.tool_calls:
                    messages.append(
                        {
                            "role": "model",
                            "parts": [
                                *(
                                    [{"type": "text", "text": message.content}]
                                    if message.content
                                    else []
                                ),
                                *(
                                    {
                                        "type": "tool_use",
                                        "id": tc.id,
                                        "name": tc.function.name,
                                        "input": tc.function.arguments,
                                    }
                                    for tc in message.tool_calls
                                ),
                            ],
                        }
                    )
                elif message.content:
                    messages.append(
                        {
                            "role": "model",
                            "parts": message.content,
                        }
                    )
            elif isinstance(message, ToolResultMessage):
                messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "parts": [{"type": "text", "text": message.content}],
                                "is_error": message.is_error,
                            }
                        ],
                    }
                )

        config_kwargs = {}
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs.pop("temperature")

        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        kwargs["generation_config"] = config_kwargs

        return system_instruction, messages, kwargs  # type: ignore

    async def _create_chat_completion(
        self, system_instruction, model: GeminiModelName, completion_kwargs
    ):
        """
        Create a chat completion using the Gemini API with retry handling.

        Params:
            completion_kwargs: Keyword arguments for an Gemini Messages API call

        Returns:
            Message: The message completion object
            float: The cost ($) of this completion
            int: Number of input tokens used
            int: Number of output tokens used
        """

        @self._retry_api_request
        async def _create_chat_completion_with_retry():
            return await asyncio.to_thread(
                self._client.GenerativeModel(
                    model_name=model, system_instruction=system_instruction
                ).generate_content,
                **completion_kwargs,
            )

        response = await _create_chat_completion_with_retry()

        cost = self._budget.update_usage_and_cost(
            model_info=GEMINI_CHAT_MODELS[model],
            input_tokens_used=response.usage_metadata.prompt_token_count,
            output_tokens_used=response.usage_metadata.candidates_token_count,
        )
        return (
            response,
            cost,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
        )

    def _parse_assistant_tool_calls(self, assistant_message) -> list[AssistantToolCall]:
        return [
            AssistantToolCall(
                id=c.function_call.name,
                type="function",
                function=AssistantFunctionCall(
                    name=c.function_call.name,
                    arguments=c.function_call.args,  # type: ignore
                ),
            )
            for c in assistant_message.candidates[0].content.parts
            if c.function_call
        ]

    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(APIConnectionError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=tenacity.after_log(self._logger, logging.DEBUG),
        )(func)

    def __repr__(self):
        return "GeminiProvider()"
