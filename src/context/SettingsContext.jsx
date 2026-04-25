import { createContext, useContext, useState, useEffect } from "react";

const SettingsContext = createContext();

export function SettingsProvider({ children }) {
  // Each setting reads from localStorage first, falls back to default if not found
  const [cameraEnabled, setCameraEnabled] = useState(() => {
    const saved = localStorage.getItem("cameraEnabled");
    return saved !== null ? JSON.parse(saved) : false;
  });

  const [selectedDeviceId, setSelectedDeviceId] = useState(() => {
    return localStorage.getItem("selectedDeviceId") || null;
  });

  const [translationFontSize, setTranslationFontSize] = useState(() => {
    const saved = localStorage.getItem("translationFontSize");
    return saved !== null ? Number(saved) : 48;
  });

  const [availableDevices, setAvailableDevices] = useState([]);

  // Persist cameraEnabled whenever it changes
  useEffect(() => {
    localStorage.setItem("cameraEnabled", JSON.stringify(cameraEnabled));
  }, [cameraEnabled]);

  // Persist selectedDeviceId whenever it changes
  useEffect(() => {
    if (selectedDeviceId) {
      localStorage.setItem("selectedDeviceId", selectedDeviceId);
    }
  }, [selectedDeviceId]);

  // Persist translationFontSize whenever it changes
  useEffect(() => {
    localStorage.setItem("translationFontSize", translationFontSize);
  }, [translationFontSize]);

  // Enumerate available camera devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        // Request permission first so device labels are populated
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        setAvailableDevices(videoDevices);
        // Only set a default device if nothing was saved previously
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Could not enumerate devices:", err);
      }
    };
    getDevices();
  }, []);

  return (
    <SettingsContext.Provider value={{
      cameraEnabled, setCameraEnabled,
      selectedDeviceId, setSelectedDeviceId,
      availableDevices,
      translationFontSize, setTranslationFontSize,
    }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  return useContext(SettingsContext);
}
