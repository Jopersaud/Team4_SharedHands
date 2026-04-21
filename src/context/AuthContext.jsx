import { createContext, useContext, useState } from "react";

// ⚠️ TEMPORARY — hardcoded logged-in state for testing without Firebase
// Replace this file with the full Firebase version when firebaseConfig is available

const AuthContext = createContext();

const MOCK_USER = {
  uid: "Q1I8KuL2iiY2gogzRE3O5DRWZ392",
  email: "crazybatman1815@gmail.com",
  accountStatus: "active",
  subscriptionTier: "free",
  subscriptionStatus: "active",
  premiumFeaturesEnabled: false,
  preferences: {
    camera: {
      defaultCamera: "front",
      fps: 30,
      resolution: "medium",
    },
    outputLanguage: "en",
    signLanguageType: "ASL",
  },
};

export function AuthProvider({ children }) {
  const [isLoggedIn, setIsLoggedIn] = useState(true);         // ← hardcoded true
  const [currentUser, setCurrentUser] = useState(MOCK_USER);  // ← hardcoded mock user

  const login = (userData) => {
    setCurrentUser(userData);
    setIsLoggedIn(true);
  };

  const logout = () => {
    setCurrentUser(null);
    setIsLoggedIn(false);
  };

  return (
    <AuthContext.Provider value={{
      isLoggedIn,
      currentUser,
      firebaseUser: null,
      loading: false,
      login,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
