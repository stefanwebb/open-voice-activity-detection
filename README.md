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