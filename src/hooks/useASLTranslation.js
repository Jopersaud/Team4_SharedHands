import { useEffect, useRef, useState, useCallback } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';

// Mirror of Python backend constants
const LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''); // LabelEncoder sorts alphabetically → A=0 … Z=25
const TRANSFORMER_CLASSES = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','SORRY','T','THANKYOU','U','V','W','X','Y','Z'];
const SMOOTHING_WINDOW = 10;
const CONFIDENCE_THRESHOLD = 0.5;
const SEQUENCE_LENGTH = 30; // frames for transformer
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

function drawLandmarks(ctx, landmarks, w, h) {
  const points = landmarks.map(lm => [lm.x * w, lm.y * h]);

  ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
  ctx.lineWidth = 2;
  for (const [start, end] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(points[start][0], points[start][1]);
    ctx.lineTo(points[end][0], points[end][1]);
    ctx.stroke();
  }

  ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/**
 * Runs ASL letter detection entirely in the browser.
 *
 * @param {object} params
 * @param {React.RefObject} params.videoRef  - ref to a react-webcam instance
 * @param {React.RefObject} params.canvasRef - ref to a <canvas> overlay element
 * @param {boolean}         params.enabled   - mirrors cameraEnabled from SettingsContext
 * @returns {{ detectedLetter: string, confidence: number, isReady: boolean }}
 */
export function useASLTranslation({ videoRef, canvasRef, enabled }) {
  const [detectedLetter, setDetectedLetter] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [motionGesture, setMotionGesture] = useState('');
  const [motionConfidence, setMotionConfidence] = useState(0);
  const [isReady, setIsReady] = useState(false);

  const handLandmarkerRef = useRef(null);
  const modelRef = useRef(null);
  const transformerModelRef = useRef(null);
  const predictionBufferRef = useRef([]); // rolling array, capped at SMOOTHING_WINDOW
  const sequenceBufferRef = useRef([]);   // rolling array, capped at SEQUENCE_LENGTH
  const frameCountRef = useRef(0);
  const rafIdRef = useRef(null);
  const lastCanvasSizeRef = useRef({ w: 0, h: 0 });

  // ── Initialise MediaPipe and TF.js model once on mount ──────────────────────
  useEffect(() => {
    let cancelled = false;

    async function init() {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'
      );

      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/hand_landmarker.task', // served from public/
          delegate: 'GPU',                          // auto-falls back to CPU
        },
        runningMode: 'VIDEO', // temporal tracking — smoother landmarks across frames
        numHands: 4,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.7,
      });

      const model = await tf.loadLayersModel('/asl_model/model.json');
      const transformerModel = await tf.loadGraphModel('/asl_transformer_model/model.json');

      if (!cancelled) {
        handLandmarkerRef.current = landmarker;
        modelRef.current = model;
        transformerModelRef.current = transformerModel;
        setIsReady(true);
      }
    }

    init().catch(err => console.error('useASLTranslation init error:', err));

    return () => { cancelled = true; };
  }, []);

  // ── Per-frame processing ────────────────────────────────────────────────────
  const processFrame = useCallback(() => {
    const video = videoRef.current?.video; // react-webcam stores <video> at .video
    const canvas = canvasRef.current;
    const landmarker = handLandmarkerRef.current;
    const model = modelRef.current;
    const transformerModel = transformerModelRef.current;

    if (!video || !canvas || !landmarker || !model || !transformerModel || video.readyState < 2) {
      rafIdRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Only reset canvas dimensions on resize (prevents flicker every frame)
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const last = lastCanvasSizeRef.current;
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
      last.w = cw;
      last.h = ch;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);

    // Compute objectFit:cover scale + offset so landmarks align with the video
    const scale = Math.max(cw / vw, ch / vh);
    const ox = (cw - vw * scale) / 2;
    const oy = (ch - vh * scale) / 2;
    const w = vw * scale;
    const h = vh * scale;

    const results = landmarker.detectForVideo(video, performance.now());

    let detectedThisFrame = null;

    if (results.landmarks && results.landmarks.length > 0) {
      // Draw all detected hands (translate canvas to cover-crop origin)
      ctx.save();
      ctx.translate(ox, oy);
      for (const handLandmarks of results.landmarks) {
        drawLandmarks(ctx, handLandmarks, w, h);
      }
      ctx.restore();

      // Use last hand — mirrors Python: results.hand_landmarks[-1]
      const lms = results.landmarks[results.landmarks.length - 1];

      // Build flat 63-feature vector [x0,y0,z0, …, x20,y20,z20]
      const flat = new Float32Array(63);
      for (let i = 0; i < 21; i++) {
        flat[i * 3]     = lms[i].x;
        flat[i * 3 + 1] = lms[i].y;
        flat[i * 3 + 2] = lms[i].z;
      }

      const inputTensor = tf.tensor2d(flat, [1, 63]);
      const prediction = model.predict(inputTensor);
      const argmax = prediction.argMax(1).dataSync()[0];
      detectedThisFrame = LABELS[argmax];
      inputTensor.dispose();
      prediction.dispose();

      // Append frame to sequence buffer for transformer
      const seq = sequenceBufferRef.current;
      seq.push(Array.from(flat));
      if (seq.length > SEQUENCE_LENGTH) seq.shift();
    }

    // Transformer: run every 30 frames when sequence buffer is full
    frameCountRef.current += 1;
    const seq = sequenceBufferRef.current;
    if (seq.length === SEQUENCE_LENGTH && frameCountRef.current % SEQUENCE_LENGTH === 0) {
      const seqTensor = tf.tensor3d([seq], [1, SEQUENCE_LENGTH, 63]);
      const transPred = transformerModel.predict(seqTensor);
      const transArgmax = transPred.argMax(1).dataSync()[0];
      const transConf = transPred.dataSync()[transArgmax];
      seqTensor.dispose();
      transPred.dispose();
      setMotionGesture(TRANSFORMER_CLASSES[transArgmax]);
      setMotionConfidence(transConf);
    }

    // Temporal smoothing — fixed-size ring buffer
    const buf = predictionBufferRef.current;
    buf.push(detectedThisFrame);
    if (buf.length > SMOOTHING_WINDOW) buf.shift();

    let displayLetter = '';
    let smoothedConfidence = 0;

    if (buf.length === SMOOTHING_WINDOW) {
      const counts = {};
      for (const p of buf) {
        if (p !== null) counts[p] = (counts[p] || 0) + 1;
      }
      const entries = Object.entries(counts);
      if (entries.length > 0) {
        const [best, count] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
        smoothedConfidence = count / SMOOTHING_WINDOW;
        if (smoothedConfidence >= CONFIDENCE_THRESHOLD) {
          displayLetter = best;
        }
      }
    }

    setDetectedLetter(displayLetter);
    setConfidence(smoothedConfidence);

    rafIdRef.current = requestAnimationFrame(processFrame);
  }, [videoRef, canvasRef]);

  // ── Start / stop the rAF loop based on enabled + isReady ───────────────────
  useEffect(() => {
    if (!isReady || !enabled) {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      if (!enabled) {
        setDetectedLetter('');
        setConfidence(0);
        setMotionGesture('');
        setMotionConfidence(0);
        predictionBufferRef.current = [];
        sequenceBufferRef.current = [];
        frameCountRef.current = 0;
      }
      return;
    }

    rafIdRef.current = requestAnimationFrame(processFrame);
    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [isReady, enabled, processFrame]);

  return { detectedLetter, confidence, motionGesture, motionConfidence, isReady };
}
