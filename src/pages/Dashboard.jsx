import { useEffect, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import Webcam from 'react-webcam';
import { useSettings } from "../context/SettingsContext";
import { useASLTranslation } from "../hooks/useASLTranslation";

export default function Dashboard() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [error, setError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [currentWord, setCurrentWord] = useState('');

  const { cameraEnabled, setCameraEnabled, selectedDeviceId, translationFontSize } = useSettings();

  const { detectedLetter, confidence, motionGesture, motionConfidence, isReady } = useASLTranslation({
    videoRef: webcamRef,
    canvasRef,
    enabled: cameraEnabled,
  });

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  // Reset camera state when camera is disabled
  useEffect(() => {
    if (!cameraEnabled) {
      setCameraReady(false);
    }
  }, [cameraEnabled]);

  // Keyboard word-building: Space = add letter, Backspace = delete, C = clear
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        if (detectedLetter) setCurrentWord(prev => prev + detectedLetter);
      } else if (e.code === 'Backspace') {
        e.preventDefault();
        setCurrentWord(prev => prev.slice(0, -1));
      } else if (e.key === 'c' || e.key === 'C') {
        setCurrentWord('');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [detectedLetter]);

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
            ) : detectedLetter ? (
              <>
                <p style={{ ...styles.translationText, fontSize: `${translationFontSize}px` }}>
                  {detectedLetter}
                </p>
                <p style={styles.confidenceText}>
                  Confidence: {(confidence * 100).toFixed(1)}%
                </p>
                {motionGesture && (
                  <p style={styles.motionText}>
                    Motion: {motionGesture} ({(motionConfidence * 100).toFixed(0)}%)
                  </p>
                )}
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
              onClick={() => { if (detectedLetter) setCurrentWord(prev => prev + detectedLetter); }}
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
              style={{ ...styles.wordBtn, backgroundColor: '#ef4444' }}
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
            <div style={styles.videoWrapper}>
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true}
                onUserMedia={() => setCameraReady(true)}
                onUserMediaError={() => setError('Camera access denied or not found.')}
                style={styles.video}
              />
              <canvas ref={canvasRef} style={styles.canvasOverlay} />
              {!cameraReady && !error && (
                <div style={styles.waitingOverlay}>Waiting for camera...</div>
              )}
              {cameraReady && !isReady && (
                <div style={styles.waitingOverlay}>Loading ASL model...</div>
              )}
            </div>
          ) : (
            /* Camera disabled — activate button from Dashboard1 */
            <div style={styles.disabledOverlay}>
              <button
                style={styles.activateButton}
                onClick={() => setCameraEnabled(true)}
              >
                Click to Activate Camera
              </button>
              <p style={styles.activateHint}>
                You can also manage camera access in Settings ⚙
              </p>
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
    background: "linear-gradient(180deg, #0ea5e9 0%, #7dd3fc 30%, #ffffff 65%, #ffffff 100%)",
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
  motionText: {
    fontSize: "14px",
    color: "#7c3aed",
    marginTop: "8px",
    fontWeight: "600",
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
    fontSize: "${translationFontSize}px",
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
  videoWrapper: {
    position: "relative",
    flex: 1,
    minHeight: 0,
    borderRadius: "10px",
    overflow: "hidden",
    backgroundColor: "#000",
  },
  video: {
    width: "100%",
    height: "100%",
    borderRadius: "10px",
    objectFit: "cover",
    display: "block",
  },
  canvasOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  },
  waitingOverlay: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    fontSize: "16px",
    color: "#ffffff",
    fontStyle: "italic",
  },
  disabledOverlay: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "14px",
    borderRadius: "10px",
    backgroundColor: "rgba(0,0,0,0.04)",
  },
  activateButton: {
    padding: "14px 32px",
    fontSize: "15px",
    fontWeight: "600",
    backgroundColor: "#0ea5e9",
    color: "#ffffff",
    border: "none",
    borderRadius: "30px",
    cursor: "pointer",
    fontFamily: "'Poppins', sans-serif",
    letterSpacing: "0.4px",
    boxShadow: "0 4px 14px rgba(14,165,233,0.4)",
  },
  activateHint: {
    fontSize: "13px",
    color: "#6b9fd4",
    fontStyle: "italic",
    margin: 0,
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
