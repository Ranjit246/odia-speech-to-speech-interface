# odia speech-to-speech interface

Prepared Odia Speech to Speech Interface using MMS (Meta AI model) and LLAMA V2 Model (https://huggingface.co/meta-llama/Llama-2-7b-hf).
The MMS model is already pre-trained with odia language, and the Large Language Model is fine tuned with small odia dataset just to test the interface. The LLM can be found here on the huggingface (https://huggingface.co/Ranjit/llama_v2_or).

The Interface is made up of three parts:
The STT Part, Which takes the odia audio Input and produces the corresponding odia text output.
The LLM Part, Which takes the odia output text from STT part, and generates some output (based on the instruction) in odia language.
The TTS Part, Which takes the odia output text from LLM part, and produces the output audio in Odia.

The three parts mentioned above can be merged together and can prodcue Speech to Speech Generative AI model. That might be computationally not that effiecent, but, can be made efficent.

For STT Part, Whisper Odia Fine tuned model can be used. Here is my odia whisper Fine tuned model: https://huggingface.co/Ranjit/odia_whisper_small_v3.0

### Disclaimer: Before using any of the work, kindly check the license of all model you are using. Do give it a try at your own risk.

Future Plans:
1) Now, the inference time is 2-3 minutes, that Inference time can be reduced.
2) Using less computational resource to deploy the models.
3) Deploy it to make an web application for realtime with own models. (If you are a app developer, reach out to me on ranjitpatro200@gmail.com)

## In order to execute the file present above feel free to reach out to me and discuss. I am also lokking for further implementation and collaboration in this idea. 
