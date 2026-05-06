.. _finetune_agent:

================================
FT-Agent for LLM Fine-Tuning
================================

FT-Agent is the RD-Agent scenario for autonomous LLM fine-tuning, introduced in
the ICML 2026 paper `FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language
Agents <https://arxiv.org/abs/2603.01712>`_.

The scenario automates benchmark-driven data processing, LLaMA-Factory training
configuration, fail-fast validation, OpenCompass evaluation, and feedback-based
iteration.

The full user guide is maintained in the repository:

`rdagent/app/finetune/llm/README.md <https://github.com/microsoft/RD-Agent/blob/main/rdagent/app/finetune/llm/README.md>`_

Minimal command after configuring the required ``FT_*`` settings:

.. code-block:: sh

   rdagent llm_finetune --base-model Qwen/Qwen2.5-7B-Instruct

Please read the full guide before running this scenario. A first run can download
large dataset/model assets and consume LLM API calls and GPU hours.
