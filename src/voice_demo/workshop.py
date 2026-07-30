from __future__ import annotations

import base64
import json
import sys
from collections.abc import Iterable

AUDIO_BUFFER_SIZE = 32_000


def local_transport():
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    return LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )


def recorder(span_processor, conversation_id: str):
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    audiobuffer = AudioBufferProcessor(num_channels=2, buffer_size=AUDIO_BUFFER_SIZE)
    if span_processor is not None:
        span_processor.attach_audio_buffer(audiobuffer, conversation_id)
    return audiobuffer


def cascading_pipeline(stt, llm, tts, span_processor, conversation_id: str):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )

    transport = local_transport()
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    audiobuffer = recorder(span_processor, conversation_id)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            audiobuffer,
            context_aggregator.assistant(),
        ]
    )
    return pipeline, audiobuffer


async def run_task(pipeline, conversation_id: str, before_run: Iterable = ()) -> None:
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,
        conversation_id=conversation_id,
    )
    await task.queue_frames(list(before_run))
    await PipelineRunner(handle_sigint=False if sys.platform == "win32" else True).run(
        task
    )


async def run_worker(pipeline, conversation_id: str, before_run: Iterable = ()) -> None:
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.workers.runner import WorkerRunner

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,
        conversation_id=conversation_id,
    )
    await worker.queue_frames(list(before_run))
    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)
    await runner.add_workers(worker)
    await runner.run()


def console_io(sample_rate: int = 24_000):
    from .audio import MicStream, SpeakerStream
    from .console import ConsoleStatus

    return (
        MicStream(sample_rate=sample_rate),
        SpeakerStream(sample_rate=sample_rate),
        ConsoleStatus(),
    )


openai_console_io = console_io


async def pump_openai_mic(connection, audio_in, ui) -> None:
    from .console import frame_level

    async for frame in audio_in.frames():
        await connection.input_audio_buffer.append(
            audio=base64.b64encode(frame).decode("ascii")
        )
        connection.record_user_audio(frame)
        ui.update_level(frame_level(frame))


async def handle_openai_realtime_event(connection, event, audio_out, ui) -> None:
    from voice_demo.openai.utils import execute_tool

    if event.type == "response.output_audio.delta":
        audio_out.write(base64.b64decode(event.delta))
    elif event.type == "input_audio_buffer.speech_started":
        audio_out.clear()
    elif event.type == "conversation.item.input_audio_transcription.completed":
        transcript = (event.transcript or "").strip()
        if transcript:
            ui.log(f"user:  {transcript}")
            await connection.response.create()
    elif event.type == "response.output_audio_transcript.done":
        transcript = (event.transcript or "").strip()
        if transcript:
            ui.log(f"agent: {transcript}")
    elif event.type == "response.done":
        tool_calls = [
            item
            for item in (event.response.output or [])
            if item.type == "function_call"
        ]
        for call in tool_calls:
            result = await execute_tool(call.name, call.arguments)
            await connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )
        if tool_calls:
            await connection.response.create()
    elif event.type == "error":
        raise RuntimeError(event.error)


async def run_openai_realtime_agent(
    *,
    session: dict,
    project: str,
    model: str,
    thread_id: str,
    trace_realtime,
    sample_rate: int = 24_000,
) -> None:
    import asyncio

    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    audio_in, audio_out, ui = openai_console_io(sample_rate)
    mic_task = None

    async with client.realtime.connect(model=model) as raw, trace_realtime(
        raw,
        is_agent_speaking=lambda: audio_out.buffered_bytes() > 0,
    ) as connection:
        await connection.session.update(session=session)
        audio_out.set_played_callback(connection.record_agent_audio)
        audio_in.start()
        audio_out.start()
        ui.log(f"[openai] thread_id={thread_id}")
        ui.log("[openai] connected. Stop/cancel this cell to end cleanly.")
        mic_task = asyncio.create_task(pump_openai_mic(connection, audio_in, ui))

        try:
            async for event in connection:
                await handle_openai_realtime_event(connection, event, audio_out, ui)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            while task is not None and task.cancelling():
                task.uncancel()
            await connection.close(code=1000, reason="notebook cell stopped")
        finally:
            if mic_task is not None:
                mic_task.cancel()
                await asyncio.gather(mic_task, return_exceptions=True)
            audio_in.stop()
            audio_out.stop()
            ui.finish()
