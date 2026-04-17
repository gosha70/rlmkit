# DGX Spark

Nvidia DGX Spark compact workstation — self-hosted inference on
prosumer Grace Blackwell hardware.

## Setup

Full setup instructions live in the **RLM Studio guide**
(`docs/rlm-studio-guide.md` in the repo; also served via
`/api/docs/rlm-studio-guide`). That document remains authoritative;
duplicating it here risks drift. This page is a placeholder so the
Cookbook surface is complete; once a Concepts or full-doc Learn
route ships, this reference will be swapped for a real in-app link.

## Quick reference

Typical configuration in RLM Studio:

| Field    | Value                        |
|----------|------------------------------|
| Backend  | `vllm` (the Spark ships vLLM)|
| Base URL | The Spark's local endpoint   |
| Model    | Whatever model you loaded    |

## Common errors

- See the rlm-studio-guide for DGX-specific troubleshooting. The
  Cookbook surface intentionally does not fork that content.
