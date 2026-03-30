import { createContext, useContext, useState, useEffect } from "react";

const SettingsContext = createContext();

export function SettingsProvider({ children }) {
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  const [availableDevices, setAvailableDevices] = useState([]);
  const [translationFontSize, setTranslationFontSize] = useState(48);

  // Enumerate available camera devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        // Request permission first so device labels are populated
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        setAvailableDevices(videoDevices);
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
