"""Notebook support helpers for the voice tracing workshop.

The notebooks should show the agent setup and LangSmith integration points.
This module hides the repetitive runtime plumbing: local mic/speaker setup,
console runners, audio recording taps, event playback, and cancellation cleanup.

Imports for optional voice SDKs stay inside functions so opening one notebook
doesn't require every backend's extra dependencies.
"""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Iterable

AUDIO_BUFFER_SIZE = 32_000


# ---------------------------------------------------------------------------
# Shared local console I/O
# ---------------------------------------------------------------------------


def console_io(sample_rate: int = 24_000):
    """Return the repo's local mic, speaker, and console status UI."""
    from .audio import MicStream, SpeakerStream
    from .console import ConsoleStatus

    return (
        MicStream(sample_rate=sample_rate),
        SpeakerStream(sample_rate=sample_rate),
        ConsoleStatus(),
    )


openai_console_io = console_io


# ---------------------------------------------------------------------------
# Pipecat helpers
# ---------------------------------------------------------------------------


def local_transport():
    """Return Pipecat's local mic/speaker transport."""
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    return LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )


def recorder(span_processor, conversation_id: str):
    """Create a stereo recorder and attach it to the LangSmith span processor."""
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    audiobuffer = AudioBufferProcessor(num_channels=2, buffer_size=AUDIO_BUFFER_SIZE)
    if span_processor is not None:
        span_processor.attach_audio_buffer(audiobuffer, conversation_id)
    return audiobuffer


def cascading_pipeline(stt, llm, tts, span_processor, conversation_id: str):
    """Build the standard Pipecat STT -> LLM -> TTS local pipeline."""
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
    """Run a Pipecat `PipelineTask` with tracing and turn tracking enabled."""
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
    """Run a Pipecat `PipelineWorker` with tracing and turn tracking enabled."""
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


# ---------------------------------------------------------------------------
# LiveKit helpers
# ---------------------------------------------------------------------------


def run_livekit_console(server) -> None:
    """Run a LiveKit `AgentServer` in local console mode with audio recording."""
    from livekit import agents

    sys.argv = [sys.argv[0], "console", "--record"]
    agents.cli.run_app(server)


# ---------------------------------------------------------------------------
# Google ADK Live helpers
# ---------------------------------------------------------------------------


async def run_google_adk_live_session(
    *,
    runner,
    adk_session,
    queue,
    tracing_plugin,
    audio_in,
    audio_out,
    ui,
    user_id: str,
    recv_sample_rate: int = 24_000,
    send_sample_rate: int = 16_000,
) -> None:
    """Run the ADK Live mic/playback loop around a preconfigured traced runner."""
    import asyncio

    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.genai import types as genai_types

    from .adk.events import LiveEvent
    from .audio import resample_pcm16
    from .console import frame_level

    audio_out.set_played_callback(tracing_plugin.record_agent_audio)
    audio_in.start()
    audio_out.start()
    ui.set_state("listening")
    stop = asyncio.Event()

    async def pump_mic() -> None:
        async for frame in audio_in.frames():
            queue.send_realtime(
                genai_types.Blob(
                    data=resample_pcm16(frame, recv_sample_rate, send_sample_rate),
                    mime_type=f"audio/pcm;rate={send_sample_rate}",
                )
            )
            ui.update_level(frame_level(frame))
            tracing_plugin.record_user_audio(frame)

    async def pump_responses() -> None:
        run_config = RunConfig(
            response_modalities=["AUDIO"],
            streaming_mode=StreamingMode.BIDI,
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            realtime_input_config=genai_types.RealtimeInputConfig(
                automatic_activity_detection=genai_types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=200,
                    silence_duration_ms=800,
                ),
            ),
        )
        try:
            async for raw_event in runner.run_live(
                user_id=user_id,
                session_id=adk_session.id,
                live_request_queue=queue,
                run_config=run_config,
            ):
                event = LiveEvent(raw_event)
                if event.interrupted:
                    audio_out.clear()
                    ui.set_state("hearing you")
                for chunk in event.audio_chunks:
                    audio_out.write(chunk)
                    ui.set_state("speaking")
                if event.final_user_transcript:
                    ui.log(f"user : {event.final_user_transcript}")
                if event.final_agent_transcript:
                    ui.log(f"agent: {event.final_agent_transcript}")
                if event.turn_complete:
                    ui.set_state("listening")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            ui.log(f"[adk] error: {exc}")
            stop.set()

    mic_task = asyncio.create_task(pump_mic())
    play_task = asyncio.create_task(pump_responses())
    try:
        await stop.wait()
    except asyncio.CancelledError:
        pass
    finally:
        queue.close()
        for task in (mic_task, play_task):
            task.cancel()
        await asyncio.gather(mic_task, play_task, return_exceptions=True)
        tracing_plugin.finalize(session_id=adk_session.id)
        audio_in.stop()
        audio_out.stop()
        ui.finish()


# ---------------------------------------------------------------------------
# OpenAI Realtime helpers
# ---------------------------------------------------------------------------


async def pump_openai_mic(connection, audio_in, ui) -> None:
    """Stream local mic frames into an OpenAI Realtime connection."""
    from .console import frame_level

    async for frame in audio_in.frames():
        await connection.input_audio_buffer.append(
            audio=base64.b64encode(frame).decode("ascii")
        )
        connection.record_user_audio(frame)
        ui.update_level(frame_level(frame))


async def handle_openai_realtime_event(connection, event, audio_out, ui) -> None:
    """Handle one OpenAI Realtime server event for the workshop agent."""
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
    """Run a traced OpenAI Realtime voice session with the notebook's wrapper."""
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
