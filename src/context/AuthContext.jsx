import { createContext, useContext, useState, useEffect } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "../firebaseConfig";

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);   // Firestore profile from Flask /login
  const [firebaseUser, setFirebaseUser] = useState(null); // Raw Firebase user object
  const [loading, setLoading] = useState(true);           // True until Firebase confirms auth state

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (fbUser) => {
      if (fbUser) {
        setFirebaseUser(fbUser);
        setIsLoggedIn(true);

        // Fetch the Firestore profile from Flask using the Firebase uid
        try {
          const response = await fetch("/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ uid: fbUser.uid }),
          });
          const data = await response.json();
          if (data.success) {
            setCurrentUser(data.user);
          }
        } catch (err) {
          console.error("Failed to fetch user profile from backend:", err);
        }
      } else {
        // User is signed out — clear everything
        setFirebaseUser(null);
        setCurrentUser(null);
        setIsLoggedIn(false);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Call this from LoginPage after Firebase sign-in + Flask /login response
  const login = (firestoreProfile) => {
    setCurrentUser(firestoreProfile);
    setIsLoggedIn(true);
  };

  const logout = async () => {
    try {
      await auth.signOut();
      // onAuthStateChanged fires automatically and clears everything
    } catch (err) {
      console.error("Logout error:", err);
    }
  };

  return (
    <AuthContext.Provider value={{
      isLoggedIn,
      currentUser,
      firebaseUser,
      loading,
      login,
      logout,
    }}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
