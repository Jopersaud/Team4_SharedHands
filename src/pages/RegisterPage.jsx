import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";

export default function RegisterPage() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  // Inject Google Font
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match");
      return;
    }

    try {
      const response = await fetch("/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password,
        }),
      });

      const data = await response.json();

      if (data.success) {
        alert("Account created successfully!");
        navigate("/");
      } else {
        alert("Error creating account: " + data.error);
      }
    } catch (error) {
      console.error("Registration error:", error);
      alert("There was an error creating your account.");
    }
  };

  return (
    <div style={styles.container}>
      <Navbar />
      <div style={styles.centered}>
        <div style={styles.card}>
          <h2 style={styles.title}>Create Account</h2>
          <form onSubmit={handleSubmit} style={styles.form}>
            <input
              name="username"
              placeholder="Username"
              value={formData.username}
              onChange={handleChange}
              required
              style={styles.input}
            />
            <input
              type="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleChange}
              required
              style={styles.input}
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleChange}
              required
              style={styles.input}
            />
            <input
              type="password"
              name="confirmPassword"
              placeholder="Confirm Password"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
              style={styles.input}
            />
            <button type="submit" style={styles.button}>
              Create Account
            </button>
          </form>
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
    background: "linear-gradient(180deg, #0ea5e9 0%, #7dd3fc 40%, #ffffff 100%)",
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
    width: "400px",
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
};
