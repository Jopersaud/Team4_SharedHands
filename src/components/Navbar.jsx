import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useSettings } from "../context/SettingsContext";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const navigate = useNavigate();

  const {
    cameraEnabled, setCameraEnabled,
    selectedDeviceId, setSelectedDeviceId,
    availableDevices,
    translationFontSize, setTranslationFontSize,
  } = useSettings();

  const { isLoggedIn, currentUser, logout } = useAuth();

  const links = [
    { label: "Home", path: "/" },
    { label: "Login", path: "/login" },
    { label: "Register", path: "/register" },
    { label: "Dashboard", path: "/dashboard" },
    // Add new pages here
  ];

  const handleLogout = async () => {
    await logout();
    setMenuOpen(false);
    navigate("/");
  };

  return (
    <>
      <div style={styles.navbar}>
        {/* Hamburger */}
        <button
          style={styles.iconButton}
          onClick={() => { setMenuOpen(!menuOpen); setSettingsOpen(false); }}
          aria-label="Toggle menu"
        >
          <span style={styles.bar} />
          <span style={styles.bar} />
          <span style={styles.bar} />
        </button>

        <span style={styles.navTitle}>SharedHands</span>

        {/* Show username if logged in */}
        {isLoggedIn && currentUser && (
          <span style={styles.userLabel}>
            {currentUser.username || currentUser.email}
          </span>
        )}

        {/* Gear icon — right-justified */}
        <button
          style={{ ...styles.iconButton, marginLeft: isLoggedIn ? "12px" : "auto", fontSize: "20px", width: "auto", height: "auto", padding: "2px 4px" }}
          onClick={() => { setSettingsOpen(!settingsOpen); setMenuOpen(false); }}
          aria-label="Settings"
        >
          ⚙
        </button>

        {/* Nav dropdown */}
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
            {/* Logout button — only shown when logged in */}
            {isLoggedIn && (
              <button
                style={{ ...styles.dropdownItem, color: "#fca5a5" }}
                onClick={handleLogout}
              >
                Log Out
              </button>
            )}
          </div>
        )}
      </div>

      {/* Settings overlay */}
      {settingsOpen && (
        <div
          style={styles.overlay}
          onClick={() => setSettingsOpen(false)}
        />
      )}

      {/* Settings drawer */}
      <div style={{
        ...styles.drawer,
        transform: settingsOpen ? "translateX(0)" : "translateX(100%)",
      }}>
        <div style={styles.drawerHeader}>
          <span style={styles.drawerTitle}>Settings</span>
          <button style={styles.closeButton} onClick={() => setSettingsOpen(false)}>✕</button>
        </div>

        {/* Camera section */}
        <div style={styles.section}>
          <p style={styles.sectionTitle}>Camera</p>

          <div style={styles.settingRow}>
            <span style={styles.settingLabel}>Camera Access</span>
            <button
              style={{
                ...styles.toggle,
                backgroundColor: cameraEnabled ? "#6ee7b7" : "#94a3b8",
              }}
              onClick={() => setCameraEnabled(!cameraEnabled)}
            >
              <span style={{
                ...styles.toggleThumb,
                transform: cameraEnabled ? "translateX(23px)" : "translateX(3px)",
              }} />
            </button>
          </div>

          {availableDevices.length > 1 && (
            <div style={styles.settingColumn}>
              <span style={styles.settingLabel}>Select Camera</span>
              <select
                style={styles.select}
                value={selectedDeviceId || ""}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
              >
                {availableDevices.map((device, i) => (
                  <option key={device.deviceId} value={device.deviceId} style={{ backgroundColor: "#1e3a8a", color: "#ffffff" }}>
                    {device.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          {availableDevices.length <= 1 && (
            <p style={styles.hint}>Only one camera detected.</p>
          )}
        </div>

        {/* Display section */}
        <div style={styles.section}>
          <p style={styles.sectionTitle}>Display</p>

          <div style={styles.settingColumn}>
            <div style={styles.settingRow}>
              <span style={styles.settingLabel}>Translation Text Size</span>
              <span style={styles.settingValue}>{translationFontSize}px</span>
            </div>
            <input
              type="range"
              min="24"
              max="96"
              step="4"
              value={translationFontSize}
              onChange={(e) => setTranslationFontSize(Number(e.target.value))}
              style={styles.slider}
            />
            <div style={styles.sliderLabels}>
              <span>Small</span>
              <span>Large</span>
            </div>
          </div>

          <div style={styles.preview}>
            <span style={{ fontSize: `${translationFontSize * 0.4}px`, fontWeight: "600", color: "#1e3a8a" }}>A</span>
            <span style={styles.previewLabel}>Preview</span>
          </div>
        </div>

        {/* Account section — only shown when logged in */}
        {isLoggedIn && (
          <div style={styles.section}>
            <p style={styles.sectionTitle}>Account</p>
            <p style={styles.settingLabel}>
              Signed in as {currentUser?.username || currentUser?.email}
            </p>
            <button
              style={styles.logoutButton}
              onClick={handleLogout}
            >
              Log Out
            </button>
          </div>
        )}
      </div>
    </>
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
  iconButton: {
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
    color: "#ffffff",
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
  userLabel: {
    marginLeft: "auto",
    fontSize: "13px",
    fontWeight: "500",
    color: "rgba(255,255,255,0.85)",
    fontFamily: "'Poppins', sans-serif",
    letterSpacing: "0.3px",
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
    zIndex: 200,
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
  overlay: {
    position: "fixed",
    inset: 0,
    backgroundColor: "rgba(0,0,0,0.3)",
    zIndex: 150,
  },
  drawer: {
    position: "fixed",
    top: 0,
    right: 0,
    width: "300px",
    height: "100vh",
    backgroundColor: "#1e3a8a",
    boxShadow: "-4px 0 20px rgba(0,0,0,0.3)",
    zIndex: 200,
    display: "flex",
    flexDirection: "column",
    transition: "transform 0.3s ease",
    fontFamily: "'Poppins', sans-serif",
    overflowY: "auto",
    borderRadius: "20px 0px 0px 20px",
  },
  drawerHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 20px 16px 20px",
    borderBottom: "1px solid rgba(255,255,255,0.15)",
  },
  drawerTitle: {
    fontSize: "18px",
    fontWeight: "600",
    color: "#ffffff",
    letterSpacing: "0.5px",
  },
  closeButton: {
    background: "none",
    border: "none",
    color: "#ffffff",
    fontSize: "18px",
    cursor: "pointer",
    padding: "2px 6px",
  },
  section: {
    padding: "20px",
    borderBottom: "1px solid rgba(255,255,255,0.1)",
  },
  sectionTitle: {
    margin: "0 0 14px 0",
    fontSize: "11px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "1px",
    color: "#7dd3fc",
  },
  settingRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: "12px",
  },
  settingColumn: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    marginBottom: "12px",
  },
  settingLabel: {
    fontSize: "14px",
    color: "#e0f2fe",
    fontWeight: "400",
  },
  settingValue: {
    fontSize: "13px",
    color: "#7dd3fc",
    fontWeight: "600",
  },
  hint: {
    fontSize: "12px",
    color: "#94a3b8",
    fontStyle: "italic",
    margin: "4px 0 0 0",
  },
  toggle: {
    width: "46px",
    height: "26px",
    borderRadius: "13px",
    border: "none",
    cursor: "pointer",
    position: "relative",
    transition: "background-color 0.2s",
    padding: 0,
    flexShrink: 0,
  },
  toggleThumb: {
    position: "absolute",
    top: "3px",
    left: "0px",
    width: "20px",
    height: "20px",
    borderRadius: "50%",
    backgroundColor: "#ffffff",
    transition: "transform 0.2s",
  },
  select: {
    width: "100%",
    padding: "8px 10px",
    borderRadius: "8px",
    border: "1px solid rgba(255,255,255,0.2)",
    backgroundColor: "#1e3a8a",
    color: "#ffffff",
    fontSize: "13px",
    fontFamily: "'Poppins', sans-serif",
    cursor: "pointer",
    outline: "none",
  },
  slider: {
    width: "100%",
    accentColor: "#6ee7b7",
    cursor: "pointer",
  },
  sliderLabels: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "11px",
    color: "#94a3b8",
  },
  preview: {
    backgroundColor: "#dbeafe",
    borderRadius: "10px",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "4px",
    marginTop: "4px",
  },
  previewLabel: {
    fontSize: "10px",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "0.8px",
  },
  logoutButton: {
    marginTop: "12px",
    width: "100%",
    padding: "10px",
    fontSize: "14px",
    fontWeight: "600",
    backgroundColor: "rgba(239,68,68,0.2)",
    color: "#fca5a5",
    border: "1px solid rgba(239,68,68,0.3)",
    borderRadius: "10px",
    cursor: "pointer",
    fontFamily: "'Poppins', sans-serif",
  },
};
