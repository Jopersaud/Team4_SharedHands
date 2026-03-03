import { useEffect, useRef, useState } from "react";

export default function Dashboard() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [error, setError] = useState(null);

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
      <h1>Dashboard</h1>

      {error ? (
        <div style={styles.error}>{error}</div>
      ) : (
        <>
          <div style={styles.videoContainer}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              style={styles.video}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
          <button onClick={captureAndUploadImage} disabled={uploading} style={styles.button}>
            {uploading ? 'Uploading...' : 'Capture and Upload'}
          </button>
          {uploadSuccess && (
            <div style={styles.success}>
              Upload successful! Image URL: <a href={imageUrl} target="_blank" rel="noopener noreferrer">{imageUrl}</a>
            </div>
          )}
          {uploadError && <div style={styles.error}>{uploadError}</div>}
        </>
        <div style={styles.videoContainer}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            style={styles.video}
          />
        </div>
      )}
    </div>
  );
}

const styles = {
  page: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    paddingTop: "20px",
    alignItems: 'center',
  },
  videoContainer: {
    width: "50%",
    height: "auto",
    backgroundColor: "black",
    display: "flex",
    marginBottom: '20px',
  },
  videoContainer: {
    width: "50%",
    height: "800px",
    backgroundColor: "black",
    display: "flex",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  button: {
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
    marginBottom: '10px',
  },
  error: {
    marginTop: "20px",
    color: "red",
    fontWeight: "bold",
  },
  success: {
    marginTop: "20px",
    color: "green",
    fontWeight: "bold",
  },
};