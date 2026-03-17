# Claude Code Voice Mode Guide

> Note: Historical copy only. The canonical copy now lives in `/home/maz3ppa/projects/voice_coding_research/docs/claude-code-workflow.md`.

## Purpose

This document describes the current hands-free workflow for talking to Claude Code through the `voice-mode` MCP server while keeping detailed technical output on screen.

The goal is not to remove the keyboard from the workflow entirely. The goal is to remove most of the typing burden so the user can:

- manually start a voice session
- keep an ongoing spoken conversation with Claude while it works
- hear only short summaries instead of long terminal output
- fall back to OS-level voice controls or the keyboard only when necessary

## Current Status

The working path is:

- Android phone microphone feeding Windows
- WSL audio bridge feeding Linux audio input/output
- Claude Code using the `voice-mode` MCP `converse` tool
- OpenAI STT for transcription
- OpenAI `gpt-4o-mini-tts` with `shimmer` as the preferred spoken voice

This path is working in Claude Code.

Important distinction:

- The standalone `uvx voice-mode converse --continuous` CLI is not a full assistant conversation loop by itself.
- The Claude Code MCP path works because Claude supplies the missing "brain" and decides when to call `converse`.

## Quick Start

1. Make sure the phone microphone is connected to Windows and still awake.
2. Start Claude Code in WSL.
3. Confirm the `voice-mode` MCP server is visible in Claude Code.
4. Paste the prompt from `docs/CLAUDE_CODE_VOICE_WORK_MODE_PROMPT.md`.
5. Let Claude speak the opening line.
6. Continue as a spoken conversation while Claude works.

## Recommended Workflow

### What Voice Mode Is Best At

- talking to Claude about the task
- asking for progress updates
- steering implementation choices
- requesting summaries
- asking for the next action
- continuing a free-form work conversation while Claude is busy

### What Should Stay On Screen

By default, Claude should not read these aloud:

- code blocks
- diffs
- logs
- stack traces
- long file paths
- large terminal outputs

The right default is:

- spoken output: short summary
- terminal output: full detail

### What Still Belongs to OS-Level Voice Control

Claude voice mode is not the same thing as full computer control.

Use Windows Voice Access or similar tools for:

- selecting text
- copy and paste
- cursor movement
- submit or enter
- switching windows
- editor control outside Claude itself

Think of the system like this:

- `voice-mode` in Claude Code: talk to the AI
- Windows Voice Access: control the computer

## Suggested Spoken Conventions

These phrases work well as lightweight in-session controls:

- `details`: allow a slightly longer explanation
- `summarize`: compress the current answer
- `next step`: tell me only what to do now
- `read code`: read only the requested code snippet
- `voice brief mode`: keep replies extra short
- `voice detail mode`: allow somewhat longer spoken replies
- `voice coding mode`: do not read code aloud unless explicitly asked

## Technical Architecture

### High-Level Flow

The current working architecture is:

1. Android phone microphone captures speech.
2. AudioRelay or a similar Windows-side input path exposes that microphone to Windows.
3. WSLg audio integration makes the Windows audio path visible inside WSL.
4. ALSA is bridged to PulseAudio in WSL through the user audio config.
5. Claude Code runs inside WSL and connects to the `voice-mode` MCP server.
6. Claude calls the `voice-mode` `converse` tool when it wants to speak and listen.
7. `voice-mode` sends TTS requests to OpenAI and plays the spoken result locally.
8. `voice-mode` records the user's reply and sends audio to OpenAI STT.
9. Claude receives the transcription result and decides what to say next.

### Why the Claude Code Path Works

The key point is that the MCP tool is only one part of the system.

`voice-mode` can:

- speak text
- listen
- transcribe speech

Claude Code adds the missing decision layer:

- it decides what to say
- it decides when to call `converse`
- it decides how much to say out loud versus leave on screen

Without Claude, `voice-mode` is not a full assistant by itself.

### Current Local Configuration

The main user-level pieces are:

- Claude Code MCP registration in `~/.claude.json`
- voice-mode configuration in `~/.voicemode/voicemode.env`
- ALSA-to-Pulse bridge in `~/.asoundrc`

The current voice preference is configured to favor:

- TTS model: `gpt-4o-mini-tts`
- preferred voice order: `shimmer`, then `nova`, then `af_sky`

## Notes on Local Fixes

During setup, a local hotfix was applied to the installed `voice-mode` CLI to prevent the standalone continuous command from sending empty strings to TTS.

Important:

- that hotfix was for the standalone CLI path
- the working Claude Code MCP workflow does not depend on that standalone CLI loop

So if the standalone `uvx voice-mode converse --continuous` command regresses later, that does not automatically mean the Claude Code MCP voice workflow is broken.

## Will This Survive Claude Code Updates?

Usually, yes, but not with an absolute guarantee.

### Likely to Survive

These parts should usually survive a normal Claude Code update:

- `~/.claude.json` MCP registration
- `~/.voicemode/voicemode.env` model and voice preferences
- `~/.asoundrc` audio bridge settings
- the general pattern of Claude calling `voice-mode` through MCP

### What Might Need Attention Later

These are the parts most likely to need rechecking:

- if Claude Code changes how MCP servers are loaded or displayed
- if `voice-mode` itself updates and changes its tool behavior
- if the local `uv` cache is cleared or rebuilt
- if OpenAI model names, API behavior, or package support change

### Important Nuance

A Claude Code update and a `voice-mode` package refresh are not the same thing.

- A Claude Code update alone will probably not break the core MCP setup.
- A `voice-mode` reinstall or cache refresh could remove local tweaks applied inside the installed package.

In practical terms:

- the Claude Code MCP workflow is fairly likely to keep working
- the standalone CLI hotfix is the most fragile part because it lives inside an installed package cache

## Maintenance Checklist After Updates

If something stops working after an update, check these in order:

1. Confirm `voice-mode` still appears as an MCP server in Claude Code.
2. Confirm the phone microphone is still available in Windows and WSL.
3. Confirm `~/.voicemode/voicemode.env` still contains the preferred TTS model and voice order.
4. Run a quick one-shot test:

```bash
uvx voice-mode converse --message "Voice check." --no-wait
```

1. If one-shot speech works but Claude does not speak, test the MCP flow inside Claude Code again.
2. If only the standalone continuous CLI is broken, assume the local CLI hotfix may need to be reapplied.

## Practical Limitations

This setup does not currently provide:

- always-on wake-word detection
- hands-free Claude session startup from silence
- full editor control through Claude alone
- perfect zero-latency spoken replies

This setup does provide:

- a workable free-form spoken Claude session
- hands-light interaction once the session is started
- concise spoken progress updates while detailed output stays on screen

## Recommended Usage Pattern

The best current pattern is:

1. Start Claude Code manually.
2. Enable or begin the voice workflow manually.
3. Paste the session prompt from `docs/CLAUDE_CODE_VOICE_WORK_MODE_PROMPT.md`.
4. Use spoken conversation to steer Claude while it works.
5. Use Windows Voice Access for computer-control tasks.
6. Use the keyboard only for exceptional recovery or precise editing.

That gives the highest benefit with the lowest complexity.
