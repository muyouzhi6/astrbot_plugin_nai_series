"""Exercise the real AstrBot agent loop, including preambles and status text."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import test_behavior as behavior
from test_behavior import Event, png


@unittest.skipIf(behavior.NaiSeriesPlugin is None, "Requires AstrBot")
class AgentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = behavior.PluginTests.asyncSetUp
    asyncTearDown = behavior.PluginTests.asyncTearDown
    finish = behavior.PluginTests.finish

    async def test_real_agent_silent_background_nonstream_and_stream(self):
        from astrbot.core.agent.hooks import BaseAgentRunHooks
        from astrbot.core.agent.run_context import ContextWrapper
        from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
        from astrbot.core.agent.tool import ToolSet
        from astrbot.core.astr_agent_run_util import run_agent
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
        from astrbot.core.provider.entities import LLMResponse, ProviderRequest
        from astrbot.core.provider.provider import Provider
        from astrbot_plugin_nai_series.providers import GeneratedImage
        from astrbot_plugin_nai_series.tools import make_tools

        plugin = self.plugin
        plugin.config["enable_llm_tool"] = True

        class ProviderStub(Provider):
            def __init__(self):
                super().__init__({}, {})
                self.calls = 0

            def get_current_key(self):
                return "test"

            def set_key(self, key):
                pass

            async def get_models(self):
                return ["test"]

            async def text_chat(self, **kwargs):
                self.calls += 1
                return LLMResponse(
                    role="assistant",
                    completion_text="正在画, 稍等.",
                    tools_call_name=["nai5_generate_image"],
                    tools_call_args=[{"prompt": "cat"}],
                    tools_call_ids=["nai-test"],
                )

            async def text_chat_stream(self, **kwargs):
                yield LLMResponse(role="assistant", completion_text="正在画", is_chunk=True)
                yield await self.text_chat(**kwargs)

        class Hooks(BaseAgentRunHooks):
            async def on_tool_start(self, context, tool, args):
                await plugin.before_tool(context.context.event, tool, args)

            async def on_tool_end(self, context, tool, args, result):
                context.context.event.clear_result()

            async def on_agent_done(self, context, response):
                await plugin.agent_done(context.context.event, context, response)

        for streaming in [False, True]:
            for status in [False, True]:
                with self.subTest(streaming=streaming, status=status):
                    event = Event()
                    event.trace = MagicMock()
                    event.is_stopped = lambda: False
                    event.get_platform_name = lambda: "aiocqhttp"
                    event.get_platform_id = lambda: "QQ"
                    event.set_result = lambda value: setattr(event, "result", value)

                    async def original_stream(generator, use_fallback=False):
                        async for chain in generator:
                            await event.send(chain)

                    event.send_streaming = original_stream
                    request = ProviderRequest(
                        prompt="画猫", contexts=[], func_tool=ToolSet(tools=make_tools(plugin))
                    )
                    await plugin.silence_tool_status(event, request)
                    gate = asyncio.Event()

                    async def generate(request):
                        await gate.wait()
                        return [GeneratedImage(png())], "test"

                    plugin.router.generate.side_effect = generate
                    provider = ProviderStub()
                    runner = ToolLoopAgentRunner()
                    await runner.reset(
                        provider=provider,
                        request=request,
                        run_context=ContextWrapper(context=SimpleNamespace(event=event)),
                        tool_executor=FunctionToolExecutor(),
                        agent_hooks=Hooks(),
                        streaming=streaming,
                    )
                    if streaming:
                        await event.send_streaming(
                            run_agent(runner, show_tool_use=status, show_tool_call_result=status)
                        )
                    else:
                        async for chain in run_agent(
                            runner, show_tool_use=status, show_tool_call_result=status
                        ):
                            await plugin.silence_tool_reply(event)
                            if event.get_result():
                                await event.send(event.get_result())
                    self.assertTrue(runner.done())
                    self.assertEqual(provider.calls, 1)
                    self.assertFalse(event.messages)
                    self.assertEqual(plugin.pending, 1)
                    gate.set()
                    await self.finish()
