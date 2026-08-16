import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { readStoredTheme } from "./theme";

// 渲染前同步套用已存主题，避免深色用户刷新时先闪一下浅色
document.documentElement.dataset.theme = readStoredTheme();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
