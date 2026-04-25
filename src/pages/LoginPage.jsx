import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { auth } from "../firebaseConfig";
import { signInWithEmailAndPassword } from "firebase/auth";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      // Step 1: Sign in with Firebase using real credentials
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const uid = userCredential.user.uid;

      // Step 2: Send uid to Flask to get full Firestore profile
      const response = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uid }),
      });

      const data = await response.json();

      if (data.success) {
        // Step 3: Store the real user profile in AuthContext
        login(data.user);
        navigate("/dashboard");
      } else {
        alert("Backend login failed: " + data.error);
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("Login failed: " + error.message);
    }
  };

  return (
    <div style={styles.container}>
      <Navbar />
      <div style={styles.centered}>
        <div style={styles.card}>
          <h2 style={styles.title}>Login</h2>
          <form onSubmit={handleLogin} style={styles.form}>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={styles.input}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={styles.input}
            />
            <button type="submit" style={styles.button}>Sign In</button>
          </form>
          <button
            style={styles.secondaryButton}
            onClick={() => navigate("/register")}
          >
            Create User
          </button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "0 24px 16px 24px",
    background: "linear-gradient(180deg, #0ea5e9 0%, #7dd3fc 30%, #ffffff 65%, #ffffff 100%)",
    fontFamily: "'Poppins', sans-serif",
    boxSizing: "border-box",
  },
  centered: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
  },
  card: {
    padding: "36px 32px",
    backgroundColor: "#dbeafe",
    borderRadius: "20px",
    width: "360px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.2)",
    textAlign: "center",
  },
  title: {
    margin: "0 0 24px 0",
    fontSize: "22px",
    fontWeight: "600",
    color: "#1e3a8a",
    letterSpacing: "0.5px",
  },
  form: {
    display: "flex",
    flexDirection: "column",
  },
  input: {
    marginBottom: "12px",
    padding: "11px 14px",
    fontSize: "15px",
    border: "1px solid #bfdbfe",
    borderRadius: "10px",
    outline: "none",
    backgroundColor: "#ffffff",
    color: "#1e3a8a",
    fontFamily: "'Poppins', sans-serif",
  },
  button: {
    padding: "12px",
    fontSize: "15px",
    fontWeight: "600",
    backgroundColor: "#6ee7b7",
    color: "#ffffff",
    border: "none",
    borderRadius: "30px",
    cursor: "pointer",
    letterSpacing: "0.4px",
    boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
    fontFamily: "'Poppins', sans-serif",
  },
  secondaryButton: {
    marginTop: "12px",
    padding: "12px",
    fontSize: "15px",
    fontWeight: "600",
    backgroundColor: "#1d4ed8",
    color: "#ffffff",
    border: "none",
    borderRadius: "30px",
    cursor: "pointer",
    letterSpacing: "0.4px",
    boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
    width: "100%",
    fontFamily: "'Poppins', sans-serif",
  },
};
