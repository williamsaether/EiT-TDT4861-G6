'use strict';

(function (global) {
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];

  class DrivingOnnxClassifier {
    constructor(opts) {
      this.modelUrl = String(opts.modelUrl || '');
      this.classNames = Array.isArray(opts.classNames) ? opts.classNames.slice() : [];
      this.excludedClasses = new Set(Array.isArray(opts.excludedClasses) ? opts.excludedClasses : []);
      this.imageSize = Number(opts.imageSize || 224);

      this.session = null;
      this.inputName = null;
      this.outputName = null;
      this.ready = false;
      this.classIndices = this.classNames
        .map((name, idx) => ({ name, idx }))
        .filter((entry) => !this.excludedClasses.has(entry.name));
    }

    async init() {
      if (!global.ort || !global.ort.InferenceSession) {
        throw new Error('onnxruntime-web is not loaded');
      }
      if (!this.modelUrl) {
        throw new Error('Missing ONNX model URL');
      }

      this.session = await global.ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      this.ready = true;
    }

    _preprocess(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const data = imageData.data;
      const pixelCount = width * height;
      const out = new Float32Array(3 * pixelCount);

      for (let i = 0; i < pixelCount; i += 1) {
        const src = i * 4;
        const r = data[src] / 255.0;
        const g = data[src + 1] / 255.0;
        const b = data[src + 2] / 255.0;

        out[i] = (r - MEAN[0]) / STD[0];
        out[pixelCount + i] = (g - MEAN[1]) / STD[1];
        out[2 * pixelCount + i] = (b - MEAN[2]) / STD[2];
      }

      return out;
    }

    _softmax(logits) {
      let maxLogit = -Infinity;
      for (let i = 0; i < logits.length; i += 1) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
      }

      const exps = new Float32Array(logits.length);
      let sumExp = 0;
      for (let i = 0; i < logits.length; i += 1) {
        const value = Math.exp(logits[i] - maxLogit);
        exps[i] = value;
        sumExp += value;
      }

      if (sumExp <= 0) {
        return exps;
      }

      for (let i = 0; i < exps.length; i += 1) {
        exps[i] /= sumExp;
      }
      return exps;
    }

    async classify(imageData) {
      if (!this.ready || !this.session || !this.inputName || !this.outputName) {
        throw new Error('Classifier is not ready');
      }

      const input = this._preprocess(imageData);
      const tensor = new global.ort.Tensor('float32', input, [1, 3, this.imageSize, this.imageSize]);
      const feeds = { [this.inputName]: tensor };
      const outputs = await this.session.run(feeds);
      const logits = outputs[this.outputName].data;
      const probs = this._softmax(logits);

      let probsSum = 0;
      for (const item of this.classIndices) {
        probsSum += probs[item.idx];
      }

      let bestLabel = 'unknown';
      let bestProb = 0;

      for (const item of this.classIndices) {
        const normalized = probsSum > 0 ? probs[item.idx] / probsSum : 0;
        if (normalized > bestProb) {
          bestProb = normalized;
          bestLabel = item.name;
        }
      }

      return {
        label: bestLabel,
        confidence: Number(bestProb.toFixed(4)),
      };
    }
  }

  global.DrivingOnnxClassifier = DrivingOnnxClassifier;
})(window);
