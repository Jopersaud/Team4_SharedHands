export default function Dashboard() {
  return (
    <div style={styles.container}>
      <h1>Dashboard</h1>

      <div style={styles.videoContainer}>
        <p style={{ color: "white" }}>Camera feed will display here</p>
        {/* Future implementation:
        <video autoPlay playsInline />
        */}
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: "40px",
  },
  videoContainer: {
    marginTop: "20px",
    width: "100%",
    height: "400px",
    backgroundColor: "black",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
};