"""Plumbing for the ElevenLabs backend, kept out of the way of the tracing.

Neither module has anything to do with LangSmith: ``audio`` is the local
mic/speaker wiring ElevenLabs' SDK requires, and ``tunnel`` is the setup that
makes a local receiver reachable from ElevenLabs' servers. The parts worth
reading are ``agent`` and ``webhook`` one level up.
"""
