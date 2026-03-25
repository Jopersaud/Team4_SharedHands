import { useEffect, useRef, useState } from "react";
import Navbar from "../components/Navbar";
import Webcam from 'react-webcam';
import io from 'socket.io-client';

const socket = io('http://100.91.247.124:5000'); 

export default function Dashboard() {
  const webcamRef = useRef(null);
  const [error, setError] = useState(null);
  const [predictedLetter, setPredictedLetter] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [processedFrame, setProcessedFrame] = useState(''); // New state for processed frame

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

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
        setProcessedFrame(data.frame); // Update processed frame
      }
    });
    
    socket.on('translation_error', (data) => {
        console.error('Translation error:', data.error);
        setError('A translation error occurred on the server.');
    });

    // Frame-sending interval
    const intervalId = setInterval(() => {
      if (socket.connected && webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit('video_frame', imageSrc);
        }
      }
    }, 200); // Send a frame every 200ms (5 FPS)

    return () => {
      console.log('Cleaning up Dashboard component');
      clearInterval(intervalId);
      socket.off('connect');
      socket.off('disconnect');
      socket.off('translation_result');
      socket.off('translation_error');
    };
  }, []);

  return (
    <div style={styles.page}>

      <Navbar />

      <div style={styles.contentRow}>
        {/* Left: Translation display */}
        <div style={styles.translationPanel}>
          <p style={styles.panelLabel}>Real-Time Translation</p>
          <div style={styles.translationBody}>
            {error ? (
              <p style={{...styles.placeholder, color: 'red'}}>{error}</p>
            ) : predictedLetter ? (
              <>
                <p style={styles.translationText}>Letter: {predictedLetter}</p>
                <p style={styles.confidenceText}>Confidence: {(confidence * 100).toFixed(2)}%</p>
              </>
            ) : (
              <p style={styles.placeholder}>Show a hand sign to the camera...</p>
            )}
          </div>
        </div>

        {/* Right: Video display */}
        <div style={styles.videoPanel}>
          <p style={styles.panelLabel}>Video Display</p>
          {/* Hidden Webcam for sending frames */}
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            style={{ display: 'none' }} // Hide the original webcam feed
          />
          {/* Display the processed frame from the backend */}
          {processedFrame && (
            <img src={processedFrame} alt="Processed Webcam Feed" style={styles.video} />
          )}
          {!processedFrame && !error && (
              <div style={{...styles.video, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6b9fd4'}}>
                  Waiting for video feed...
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
    background: "linear-gradient(180deg, #0ea5e9 0%, #38bdf8 50%, #7dd3fc 100%)",
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
    fontSize: "48px",
    fontWeight: "600",
    color: "#1e3a8a",
    margin: "0 0 10px 0",
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
  videoPanel: {
    flex: "1 1 75%",
    backgroundColor: "#dbeafe",
    borderRadius: "16px",
    padding: "16px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  video: {
    width: "100%",
    flex: 1,
    borderRadius: "10px",
    objectFit: "cover",
    backgroundColor: "#000",
    minHeight: 0,
    transform: "scaleX(-1)", // Mirror the video
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
