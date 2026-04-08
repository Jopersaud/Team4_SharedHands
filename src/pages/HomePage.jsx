import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";

export default function HomePage() {
  const navigate = useNavigate();

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  return (
    <div style={styles.page}>
      <Navbar />

      <div style={styles.body}>
        {/* Welcome text */}
        <div style={styles.heroText}>
          <p style={styles.welcomeLine}>Welcome to</p>
          <p style={styles.appName}>SharedHands</p>
        </div>

        {/* Buttons */}
        <div style={styles.buttonRow}>
          <button
            style={styles.primaryButton}
            onClick={() => navigate("/register")}
          >
            Create Account
          </button>
          <button
            style={styles.secondaryButton}
            onClick={() => navigate("/login")}
          >
            Login
          </button>
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
  body: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: "80px",
    width: "100%",
    maxWidth: "1400px",
  },
  heroText: {
    marginTop: "80px",
    textAlign: "center",
  },
  welcomeLine: {
    margin: 0,
    fontSize: "28px",
    fontWeight: "400",
    color: "#1e3a8a",
  },
  appName: {
    margin: "6px 0 0 0",
    fontSize: "48px",
    fontWeight: "700",
    color: "#1e3a8a",
    letterSpacing: "1px",
  },
  buttonRow: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "center",
    gap: "80px",
    width: "100%",
  },
  primaryButton: {
    padding: "14px 40px",
    fontSize: "16px",
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
  secondaryButton: {
    padding: "14px 40px",
    fontSize: "16px",
    fontWeight: "600",
    backgroundColor: "#ffffff",
    color: "#1e3a8a",
    border: "2px solid #1e3a8a",
    borderRadius: "30px",
    cursor: "pointer",
    fontFamily: "'Poppins', sans-serif",
    letterSpacing: "0.4px",
    boxShadow: "0 4px 14px rgba(0,0,0,0.08)",
  },
};
