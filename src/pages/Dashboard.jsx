import { useEffect, useRef, useState } from "react";

export default function Dashboard() {
  const videoRef = useRef(null);
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

  return (
    <div style={styles.page}>
      <h1>Dashboard</h1>

      {error ? (
        <div style={styles.error}>{error}</div>
      ) : (
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
  error: {
    marginTop: "20px",
    color: "red",
    fontWeight: "bold",
  },
};