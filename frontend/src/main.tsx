import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Note: StrictMode intentionally off. The fleet replay endpoint is expensive
// (~12s on first paint for 42 meters), and StrictMode's dev-mode double-mount
// triggers a second concurrent POST which the browser then abandons
// (ERR_NETWORK_IO_SUSPENDED) before state can settle.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
