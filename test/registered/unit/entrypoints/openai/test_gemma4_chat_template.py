import json
import unittest
from pathlib import Path
from typing import ClassVar

import jinja2
from jinja2.ext import LoopControlExtension
from jinja2.sandbox import ImmutableSandboxedEnvironment

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
TEMPLATE_PATH = REPOSITORY_ROOT / "examples/chat_template/gemma4_chat_template.jinja"


class Gemma4ChatTemplateTest(unittest.TestCase):
    """Tests the canonical Gemma 4 chat serialization contract."""

    template: ClassVar[jinja2.Template]

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the template with the serving-time JSON helper."""
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[LoopControlExtension],
        )
        environment.globals["raise_exception"] = cls._raise_template_error
        cls.template = environment.from_string(
            TEMPLATE_PATH.read_text(encoding="utf-8")
        )

    @staticmethod
    def _raise_template_error(message: str) -> None:
        """Raise a template error from a template assertion.

        :param message: Assertion failure message.
        :raises jinja2.TemplateError: Always.
        """
        raise jinja2.TemplateError(message)

    def _render(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> str:
        """Render a chat history with generation priming enabled.

        :param messages: OpenAI-compatible message dictionaries.
        :param tools: Optional OpenAI-compatible tool declarations.
        :returns: Fully serialized prompt.
        """
        return self.template.render(
            messages=messages,
            tools=tools,
            bos_token="<bos>",
            add_generation_prompt=True,
            enable_thinking=True,
            fromjson=json.loads,
        )

    def test_terminal_assistant_role_is_preserved(self) -> None:
        """Serialize a completed assistant message as a model turn."""
        rendered = self._render(
            [
                {"role": "user", "content": "Count from one to three."},
                {"role": "assistant", "content": "One, two, three."},
            ]
        )

        self.assertEqual(
            rendered,
            "<bos><|turn>system\n"
            "<|think|>\n"
            "<turn|>\n"
            "<|turn>user\n"
            "Count from one to three.<turn|>\n"
            "<|turn>model\n"
            "One, two, three.<turn|>\n"
            "<|turn>model\n",
        )

    def test_tool_results_are_parsed_sorted_and_reasoning_is_retained(self) -> None:
        """Canonicalize JSON tool results and retain terminal tool reasoning."""
        rendered = self._render(
            [
                {"role": "user", "content": "Check Toronto."},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Fetch the current observation.",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Toronto"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": '{"temperature_c":21,"condition":"clear"}',
                },
            ]
        )

        self.assertIn(
            "<|channel>thought\nFetch the current observation.<channel|>", rendered
        )
        self.assertIn(
            '<|tool_call>call:get_weather{city:<|"|>Toronto<|"|>}<tool_call|>',
            rendered,
        )
        self.assertIn(
            '<|tool_response>response:get_weather{condition:<|"|>clear<|"|>,temperature_c:21}<tool_response|>',
            rendered,
        )
        self.assertNotIn("{value:", rendered)


if __name__ == "__main__":
    unittest.main()
