// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// TODO: Add your own Firebase configuration here
const firebaseConfig = {
  apiKey: "AIzaSyCLlLmHRfc6aWmAJSmCqTRHjv-lNcKn9fg",
  authDomain: "sharedhands-17f7b.firebaseapp.com",
  projectId: "sharedhands-17f7b",
  storageBucket: "sharedhands-17f7b.firebasestorage.app",
  messagingSenderId: "510739287577",
  appId: "1:510739287577:web:1e30c5e7249812a6882488",
  measurementId: "G-J3K5QLR74C"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { app, auth };

