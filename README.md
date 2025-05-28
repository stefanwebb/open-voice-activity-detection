<!--
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

-->

# Open Voice Activity Detection

Voice Activity Detection (VAD) is one of the first and most important steps an app with speech recognition has to perform. And yet, in researching real-time voice chat agents, I discovered that SoTA models are not truely open-source, only "open-weights." This poses challenges for researching and improving the SoTA in VAD.

In this repo, I plan to include:

* a full implementation of Silero VAD for research purposes
* code to train Silero VAD from scratch on custom data
* evaluations on common benchmarks
* LoRA fine-tuning for Silero VAD
* example integrations with Python, client-side web apps, and Unity

This repo is distributed under a very permissive license, [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/deed.en) to encourage both research *and* commercial use.

# Fully Open-Source Voice Activity Detection (VAD) for Real-Time Speech Applications
Voice Activity Detection (VAD) is a critical first step in any application involving speech recognition. However, while exploring real-time voice chat agents, I found that many state-of-the-art (SoTA) models are not truly open-source—they provide only open weights, limiting transparency and hindering research and development.

This repository aims to change that by providing a fully open and research-friendly implementation of the Silero VAD model. The goal is to advance the state of the art in VAD through open experimentation, training, and integration.

## Status
As of May 27, 2025, this repository includes:
✅ A complete implementation of the Silero VAD model for research use

## Roadmap
In the near future, I plan to add the following:
🧠 Code to train Silero VAD from scratch on custom datasets

📊 Evaluation scripts for standard VAD benchmarks

🔧 Support for LoRA fine-tuning to extend or adapt Silero VAD

🔌 Example integrations with Python, client-side web applications, and Unity

## License

This project is released under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0), encouraging both academic research and commercial application.