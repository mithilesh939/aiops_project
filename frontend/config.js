// Runtime configuration of the UI.  The UI talks to the backend ONLY through this REST base URL
// (loose coupling).  In Docker Compose this file is replaced via a volume / env substitution.
window.QUIZPULSE_CONFIG = {
  API_BASE_URL: "http://localhost:8000",
  MLFLOW_URL: "http://localhost:5000",
  GRAFANA_URL: "http://localhost:3000",
  PROMETHEUS_URL: "http://localhost:9090",
  REFRESH_SECONDS: 5,
};
