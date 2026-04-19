import {
  pipeline,
  TextStreamer,
} from "https://esm.sh/@huggingface/transformers@4.1.0";

const MODEL_ID = "onnx-community/SmolLM2-360M-Instruct-ONNX";
const MODEL_DTYPE = "q4";

let generator = null;

const isIOS = /iPhone|iPad|iPod/.test(self.navigator?.userAgent ?? "");

async function loadModel() {
  if (generator) return;
  generator = await pipeline("text-generation", MODEL_ID, {
    device: isIOS ? "wasm" : "webgpu",
    dtype: MODEL_DTYPE,
    progress_callback: (progress) => {
      self.postMessage({ type: "progress", progress });
    },
  });
}

self.onmessage = async (e) => {
  const { type, input } = e.data;

  if (type === "load") {
    try {
      await loadModel();
      self.postMessage({ type: "ready" });
    } catch (err) {
      // fallback to wasm if WebGPU unavailable
      try {
        generator = await pipeline("text-generation", MODEL_ID, {
          device: "wasm",
          dtype: MODEL_DTYPE,
          progress_callback: (progress) => {
            self.postMessage({ type: "progress", progress });
          },
        });
        self.postMessage({ type: "ready" });
      } catch (err2) {
        self.postMessage({ type: "error", message: err2.message });
      }
    }
  }

  if (type === "generate") {
    try {
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
          self.postMessage({ type: "token", text });
        },
      });

      const messages = [
        {
          role: "user",
          content: "What's on my mind: I feel stuck and uninspired.",
        },
        {
          role: "assistant",
          content: "1. open a window and sketch the first thing you see outside\n2. go for a 10 minute walk and notice textures\n3. draw a nearby object from an unusual angle",
        },
        {
          role: "user",
          content: `What's on my mind: ${input}`,
        },
      ];

      const output = await generator(messages, {
        max_new_tokens: 80,
        do_sample: false,
        repetition_penalty: 1.3,
        no_repeat_ngram_size: 3,
        streamer,
      });

      const fullText = output[0].generated_text.at(-1).content;
      self.postMessage({ type: "complete", text: fullText });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
  }
};
