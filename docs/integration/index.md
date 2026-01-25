# Integration Guide

How to integrate Datasculpt into your data pipeline.

## What You Need to Integrate

| Component | Required? | Purpose |
|-----------|-----------|---------|
| Core inference | Yes | Shape, grain, roles |
| Interactive mode | Optional | Resolve ambiguity |
| Optional adapters | Optional | Enhanced profiling |
| Invariant handoff | Optional | Downstream registration |

## Recommended Order

1. **Start minimal** — Core inference only
2. **Add interactive mode** — When ambiguity handling is needed
3. **Add adapters** — When you need richer profiling
4. **Connect to Invariant** — For full governance

## Integration Guides

| Guide | Use Case |
|-------|----------|
| [Minimal Integration](minimal-integration.md) | First integration, proof of concept |
| [Interactive Mode](interactive-mode.md) | Human-in-the-loop workflows |
| [Optional Adapters](optional-adapters.md) | Enhanced profiling capabilities |
| [Invariant Handoff](invariant-handoff.md) | Connecting to downstream governance |
