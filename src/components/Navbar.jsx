import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  const links = [
    { label: "Login", path: "/" },
    { label: "Register", path: "/register" },
    { label: "Dashboard", path: "/dashboard" },
    // Add new pages here
  ];

  return (
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
      <span style={styles.navTitle}>Translation App</span>

      {menuOpen && (
        <div style={styles.dropdown}>
          {links.map((link) => (
            <button
              key={link.path}
              style={styles.dropdownItem}
              onClick={() => { navigate(link.path); setMenuOpen(false); }}
            >
              {link.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  navbar: {
    position: "relative",
    width: "100%",
    display: "flex",
    alignItems: "center",
    padding: "12px 0",
    marginBottom: "12px",
    borderBottom: "1px solid rgba(255,255,255,0.2)",
    flexShrink: 0,
    boxSizing: "border-box",
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
    fontFamily: "'Poppins', sans-serif",
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
    background: "none",
    border: "none",
    borderBottom: "1px solid rgba(255,255,255,0.1)",
    fontSize: "14px",
    fontWeight: "500",
    fontFamily: "'Poppins', sans-serif",
    cursor: "pointer",
    textAlign: "left",
  },
};
