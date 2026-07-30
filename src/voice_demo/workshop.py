from __future__ import annotations

import base64
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


def openai_console_io(sample_rate: int = 24_000):
    from .audio import MicStream, SpeakerStream
    from .console import ConsoleStatus

    return (
        MicStream(sample_rate=sample_rate),
        SpeakerStream(sample_rate=sample_rate),
        ConsoleStatus(),
    )


async def pump_openai_mic(connection, audio_in, ui) -> None:
    from .console import frame_level

    async for frame in audio_in.frames():
        await connection.input_audio_buffer.append(
            audio=base64.b64encode(frame).decode("ascii")
        )
        connection.record_user_audio(frame)
        ui.update_level(frame_level(frame))
