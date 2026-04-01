import { useEffect, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import Webcam from 'react-webcam';
import io from 'socket.io-client';
import { useSettings } from "../context/SettingsContext";

const socket = io('http://localhost:5000', { transports: ['websocket'] });

export default function Dashboard() {
  const webcamRef = useRef(null);
  const [error, setError] = useState(null);
  const [predictedLetter, setPredictedLetter] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [processedFrame, setProcessedFrame] = useState('');
  const [cameraReady, setCameraReady] = useState(false);
  const [currentWord, setCurrentWord] = useState('');

  const { cameraEnabled, selectedDeviceId, translationFontSize } = useSettings();

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  // Reset processed frame when camera is disabled
  useEffect(() => {
    if (!cameraEnabled) {
      setProcessedFrame('');
      setCameraReady(false);
      setPredictedLetter('');
      setConfidence(0);
    }
  }, [cameraEnabled]);

  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to backend');
      setError(null);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setError('Connection to server lost.');
    });

    socket.on('translation_result', (data) => {
      setPredictedLetter(data.letter);
      setConfidence(data.confidence);
      if (data.frame) {
        setProcessedFrame(data.frame);
      }
    });

    socket.on('translation_error', (data) => {
      console.error('Translation error:', data.error);
      setError('A translation error occurred on the server.');
    });

    // Frame-sending interval — only sends if camera is enabled
    const intervalId = setInterval(() => {
      if (socket.connected && webcamRef.current && cameraEnabled) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit('video_frame', imageSrc);
        }
      }
    }, 500);

    return () => {
      clearInterval(intervalId);
      socket.off('connect');
      socket.off('disconnect');
      socket.off('translation_result');
      socket.off('translation_error');
    };
  }, [cameraEnabled]);

  // Keyboard word-building: Space = add letter, Backspace = delete, C = clear
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        if (predictedLetter) setCurrentWord(prev => prev + predictedLetter);
      } else if (e.code === 'Backspace') {
        e.preventDefault();
        setCurrentWord(prev => prev.slice(0, -1));
      } else if (e.key === 'c' || e.key === 'C') {
        setCurrentWord('');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [predictedLetter]);

  return (
    <div style={styles.page}>
      <Navbar />

      <div style={styles.contentRow}>
        {/* Left: Translation display */}
        <div style={styles.translationPanel}>
          <p style={styles.panelLabel}>Real-Time Translation</p>
          <div style={styles.translationBody}>
            {!cameraEnabled ? (
              <p style={styles.placeholder}>Camera is disabled.</p>
            ) : error ? (
              <p style={{ ...styles.placeholder, color: 'red' }}>{error}</p>
            ) : predictedLetter ? (
              <>
                {/* Font size driven by settings context */}
                <p style={{ ...styles.translationText, fontSize: `${translationFontSize}px` }}>
                  {predictedLetter}
                </p>
                <p style={styles.confidenceText}>
                  Confidence: {(confidence * 100).toFixed(1)}%
                </p>
              </>
            ) : (
              <p style={styles.placeholder}>Show a hand sign to the camera...</p>
            )}
          </div>

          {/* Word buffer */}
          <div style={styles.wordDisplay}>
            <p style={styles.wordLabel}>Word</p>
            <p style={styles.wordText}>{currentWord || '\u00A0'}</p>
          </div>

          {/* Word-building buttons */}
          <div style={styles.buttonRow}>
            <button
              style={styles.wordBtn}
              onClick={() => { if (predictedLetter) setCurrentWord(prev => prev + predictedLetter); }}
              title="Space — add current letter"
            >
              + Letter
            </button>
            <button
              style={styles.wordBtn}
              onClick={() => setCurrentWord(prev => prev.slice(0, -1))}
              title="Backspace"
            >
              ← Back
            </button>
            <button
              style={{...styles.wordBtn, backgroundColor: '#ef4444'}}
              onClick={() => setCurrentWord('')}
              title="Clear word"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right: Video display */}
        <div style={styles.videoPanel}>
          <p style={styles.panelLabel}>Video Display</p>

          {cameraEnabled ? (
            <>
              {/* Webcam — uses selectedDeviceId from settings if available */}
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true}
                onUserMedia={() => setCameraReady(true)}
                onUserMediaError={() => setError('Camera access denied or not found.')}
                style={{
                  ...styles.video,
                  display: processedFrame ? 'none' : 'block',
                }}
              />

              {/* Processed frame from backend */}
              {processedFrame && (
                <img
                  src={processedFrame}
                  alt="Processed Webcam Feed"
                  style={styles.video}
                />
              )}

              {/* Waiting message */}
              {!cameraReady && !error && (
                <div style={styles.waitingOverlay}>
                  Waiting for camera...
                </div>
              )}
            </>
          ) : (
            <div style={styles.disabledOverlay}>
              Camera is disabled. Enable it in Settings ⚙
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    height: "100vh",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 24px 16px 24px",
    background: "linear-gradient(180deg, #0ea5e9 0%, #7dd3fc 40%, #ffffff 100%)",
    fontFamily: "'Poppins', sans-serif",
    boxSizing: "border-box",
  },
  contentRow: {
    display: "flex",
    flexDirection: "row",
    gap: "16px",
    width: "100%",
    maxWidth: "1400px",
    flex: 1,
    minHeight: 0,
  },
  translationPanel: {
    flex: "0 0 25%",
    backgroundColor: "#dbeafe",
    borderRadius: "16px",
    padding: "16px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  translationBody: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    textAlign: "center",
  },
  translationText: {
    fontWeight: "600",
    color: "#1e3a8a",
    margin: "0 0 10px 0",
    transition: "font-size 0.2s ease",
  },
  confidenceText: {
    fontSize: "20px",
    color: "#1d4ed8",
    margin: 0,
  },
  placeholder: {
    fontSize: "16px",
    color: "#6b9fd4",
    fontStyle: "italic",
    margin: 0,
  },
  wordDisplay: {
    backgroundColor: "#eff6ff",
    borderRadius: "10px",
    padding: "10px 14px",
    marginTop: "12px",
    minHeight: "48px",
    border: "1px solid #bfdbfe",
    flexShrink: 0,
  },
  wordLabel: {
    fontSize: "11px",
    fontWeight: "600",
    textTransform: "uppercase",
    color: "#1d4ed8",
    margin: "0 0 4px 0",
    letterSpacing: "0.6px",
  },
  wordText: {
    fontSize: "28px",
    fontWeight: "600",
    color: "#1e3a8a",
    margin: 0,
    letterSpacing: "3px",
    wordBreak: "break-all",
  },
  buttonRow: {
    display: "flex",
    gap: "8px",
    marginTop: "10px",
    flexShrink: 0,
  },
  wordBtn: {
    flex: 1,
    padding: "8px 4px",
    borderRadius: "8px",
    border: "none",
    backgroundColor: "#1d4ed8",
    color: "#ffffff",
    fontFamily: "'Poppins', sans-serif",
    fontWeight: "600",
    fontSize: "12px",
    cursor: "pointer",
  },
  videoPanel: {
    flex: "1 1 75%",
    backgroundColor: "#dbeafe",
    borderRadius: "16px",
    padding: "16px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    position: "relative",
  },
  video: {
    width: "100%",
    flex: 1,
    borderRadius: "10px",
    objectFit: "cover",
    backgroundColor: "#000",
    minHeight: 0,
  },
  waitingOverlay: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    fontSize: "16px",
    color: "#6b9fd4",
    fontStyle: "italic",
  },
  disabledOverlay: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "15px",
    color: "#6b9fd4",
    fontStyle: "italic",
    textAlign: "center",
    borderRadius: "10px",
    backgroundColor: "rgba(0,0,0,0.04)",
  },
  panelLabel: {
    margin: "0 0 10px 0",
    fontSize: "13px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.8px",
    color: "#1d4ed8",
    flexShrink: 0,
  },
};
