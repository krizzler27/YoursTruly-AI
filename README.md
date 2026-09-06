# YoursTruly AI

**Local chat. No cloud. Runs entirely on-device**

> Under active development.

## Core Idea

Small models are sufficient for personal assistance when the orchestration around them is strong. Models are the fixed, constrained ingredient; the engineering around them handles context, continuity, and reliability. Everything runs locally and no data leaves the device.

> Baseline is 8GB RAM with no GPU and it scales per system, not scoped only to baseline. Model optimization handled automatically based on system hardware specifications.

## What It Does

A private chat that runs fully on-device with persistent history and streaming replies.

## Features

- Model Marketplace
- One-Click Download
- Hardware Inspect

## How It Runs

Built on `llama.cpp` via `llama-cpp-python` for in-process inference, offline by default, auto-tuned for low-end hardware.

## Stack

| Technology | Used For |
| --- | --- |
| Python 3.11 | Backend language |
| FastAPI | API and streaming |
| llama.cpp | Local inference |
| SQLite | Conversation storage |
| Vanilla JS | Frontend |

## Status

Still under development. Expect changes.