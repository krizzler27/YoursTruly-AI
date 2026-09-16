# YoursTruly AI

**Private assistant. No cloud. Runs entirely on-device.**

> Under active development.

## Core Idea

Small models are sufficient for personal assistance when the orchestration around them is strong. Models are the fixed, constrained ingredient and the engineering around them handles context, continuity, and reliability. Everything runs locally and no data leaves the device.

> Designed to run on 8GB RAM with no GPU. Scales per system. Better hardware, better run. Adapts automatically to your device.

## What It Does

A private assistant for everyday help that runs entirely on-device.

## Features

- Private Chat with streaming replies
- Persistent Memory that stays on device
- Private Documents - attach txt, md, pdf per chat
- Model Marketplace with hardware-aware filtering
- One-click Model download
- Hardware Inspect 

## How It Runs

Built on `llama.cpp` via `llama-cpp-python` for in-process inference, offline by default and auto-tuned for the device it runs on.

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