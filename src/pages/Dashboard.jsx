import { useEffect, useRef, useState } from "react";

export default function Dashboard() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [translationText, setTranslationText] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        if (err.name === "NotAllowedError") {
          setError("Camera access denied. Please enable permissions.");
        } else if (err.name === "NotFoundError") {
          setError("No camera device found.");
        } else {
          setError("Error accessing camera.");
        }
      }
    };

    startCamera();
  }, []);

  const captureAndUploadImage = () => {
    if (!videoRef.current) return;

    setUploading(true);
    setUploadSuccess(false);
    setUploadError(null);
    setImageUrl('');

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    canvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append('image', blob, 'capture.jpg');

      fetch('/upload-image', {
        method: 'POST',
        body: formData,
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            setUploadSuccess(true);
            setImageUrl(data.url);
            setTranslationText(data.translation || '');
          } else {
            setUploadError(data.error || 'Unknown error during upload.');
          }
        })
        .catch(err => {
          setUploadError(`Upload failed: ${err.message}`);
        })
        .finally(() => {
          setUploading(false);
        });
    }, 'image/jpeg');
  };

  return (
    <div style={styles.page}>

      {/* Navbar */}
      <div style={styles.navbar}>
        <button
          style={styles.hamburger}
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span style={styles.bar} />
          <span style={styles.bar} />
          <span style={styles.bar} />
        </button>
        <span style={styles.navTitle}>Translation Page</span>

        {/* Dropdown menu */}
        {menuOpen && (
          <div style={styles.dropdown}>
            <a href="#" style={styles.dropdownItem}>Home</a>
            <a href="#" style={styles.dropdownItem}>Settings</a>
            <a href="#" style={styles.dropdownItem}>About</a>
          </div>
        )}
      </div>

      {error ? (
        <div style={styles.error}>{error}</div>
      ) : (
        <>
          {/* Main content row */}
          <div style={styles.contentRow}>
            {/* Left: Translation display */}
            <div style={styles.translationPanel}>
              <p style={styles.panelLabel}>Translation Display</p>
              <div style={styles.translationBody}>
                {translationText
                  ? <p style={styles.translationText}>{translationText}</p>
                  : <p style={styles.placeholder}>Translation will appear here after capturing an image.</p>
                }
              </div>
            </div>

            {/* Right: Video display */}
            <div style={styles.videoPanel}>
              <p style={styles.panelLabel}>Video Display</p>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={styles.video}
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
            </div>
          </div>

          {/* Capture button centered below */}
          <div style={styles.buttonRow}>
            <button
              onClick={captureAndUploadImage}
              disabled={uploading}
              style={{
                ...styles.button,
                ...(uploading ? styles.buttonDisabled : {}),
              }}
            >
              {uploading ? 'Uploading...' : 'Capture and Upload'}
            </button>
          </div>

          {/* Status messages */}
          {uploadSuccess && (
            <div style={styles.success}>
              Upload successful!{' '}
              <a href={imageUrl} target="_blank" rel="noopener noreferrer" style={styles.link}>
                {imageUrl}
              </a>
            </div>
          )}
          {uploadError && <div style={styles.error}>{uploadError}</div>}
        </>
      )}
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 32px 24px 32px",
    background: "linear-gradient(180deg, #0ea5e9 0%, #38bdf8 50%, #7dd3fc 100%)",
    fontFamily: "'Poppins', sans-serif",
    boxSizing: "border-box",
  },

  /* Navbar */
  navbar: {
    position: "relative",
    width: "100%",
    display: "flex",
    alignItems: "center",
    padding: "14px 0",
    marginBottom: "20px",
    borderBottom: "1px solid rgba(255,255,255,0.2)",
  },
  hamburger: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-between",
    width: "28px",
    height: "20px",
    background: "none",
    border: "none",
    cursor: "pointer",
    padding: 0,
    flexShrink: 0,
  },
  bar: {
    display: "block",
    width: "100%",
    height: "3px",
    backgroundColor: "#ffffff",
    borderRadius: "2px",
  },
  navTitle: {
    marginLeft: "18px",
    fontSize: "18px",
    fontWeight: "600",
    color: "#ffffff",
    letterSpacing: "0.5px",
  },
  dropdown: {
    position: "absolute",
    top: "100%",
    left: 0,
    backgroundColor: "#1e3a8a",
    borderRadius: "10px",
    boxShadow: "0 6px 20px rgba(0,0,0,0.3)",
    display: "flex",
    flexDirection: "column",
    minWidth: "160px",
    zIndex: 100,
    overflow: "hidden",
  },
  dropdownItem: {
    padding: "12px 20px",
    color: "#ffffff",
    textDecoration: "none",
    fontSize: "14px",
    fontWeight: "500",
    borderBottom: "1px solid rgba(255,255,255,0.1)",
  },

  /* Two-panel row */
  contentRow: {
    display: "flex",
    flexDirection: "row",
    gap: "20px",
    width: "100%",
    maxWidth: "1400px",
    flex: 1,
  },

  /* Left translation panel */
  translationPanel: {
    flex: "0 0 25%",
    backgroundColor: "#dbeafe",
    borderRadius: "16px",
    padding: "20px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    minHeight: "520px",
  },
  translationBody: {
    flex: 1,
    overflowY: "auto",
  },
  translationText: {
    fontSize: "16px",
    color: "#1e3a8a",
    lineHeight: "1.6",
    margin: 0,
  },
  placeholder: {
    fontSize: "14px",
    color: "#6b9fd4",
    fontStyle: "italic",
    margin: 0,
  },

  /* Right video panel */
  videoPanel: {
    flex: "1 1 75%",
    backgroundColor: "#dbeafe",
    borderRadius: "16px",
    padding: "20px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    minHeight: "520px",
  },
  video: {
    width: "100%",
    flex: 1,
    borderRadius: "10px",
    objectFit: "cover",
    backgroundColor: "#000",
    minHeight: "400px",
  },

  /* Shared panel label */
  panelLabel: {
    margin: "0 0 12px 0",
    fontSize: "13px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.8px",
    color: "#1d4ed8",
  },

  /* Button row */
  buttonRow: {
    display: "flex",
    justifyContent: "center",
    marginTop: "20px",
    width: "100%",
    maxWidth: "1400px",
  },
  button: {
    padding: "12px 36px",
    fontSize: "15px",
    fontWeight: "600",
    cursor: "pointer",
    backgroundColor: "#6ee7b7",
    color: "#ffffff",
    border: "none",
    borderRadius: "30px",
    boxShadow: "0 4px 14px rgba(0,0,0,0.25)",
    letterSpacing: "0.4px",
  },
  buttonDisabled: {
    backgroundColor: "#555",
    cursor: "not-allowed",
    boxShadow: "none",
  },

  /* Status */
  error: {
    marginTop: "16px",
    color: "#fca5a5",
    fontWeight: "bold",
    fontSize: "14px",
  },
  success: {
    marginTop: "16px",
    color: "#6ee7b7",
    fontWeight: "bold",
    fontSize: "14px",
  },
  link: {
    color: "#6ee7b7",
    wordBreak: "break-all",
  },
};
