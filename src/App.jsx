import { Routes, Route } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import Dashboard from "./pages/Dashboard";
import { SettingsProvider } from "./context/SettingsContext";

export default function App() {
  return (
    <SettingsProvider>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
    </Routes>
    </SettingsProvider>
    
  );
}
