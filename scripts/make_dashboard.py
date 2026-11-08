"""Generate the Grafana dashboard JSON (kept as code so it is reviewable and reproducible).

python scripts/make_dashboard.py  ->  docker/grafana/dashboards/quizpulse.json
"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prom"}
_id = [0]


def target(expr, legend=""):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": "A"}


def panel(kind, title, exprs, x, y, w=8, h=7, unit=None, thresholds=None, minv=None, maxv=None, desc=""):
    _id[0] += 1
    t = []
    for i, (e, leg) in enumerate(exprs):
        tt = target(e, leg)
        tt["refId"] = chr(65 + i)
        t.append(tt)
    defaults = {"unit": unit or "short", "custom": {}}
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["color"] = {"mode": "thresholds"}
    if minv is not None:
        defaults["min"] = minv
    if maxv is not None:
        defaults["max"] = maxv
    p = {"id": _id[0], "type": kind, "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": t,
         "fieldConfig": {"defaults": defaults, "overrides": []}, "options": {}}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background", "graphMode": "area"}
    if kind == "timeseries":
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
    return p


green, red, amber = "green", "red", "orange"
panels = [
    panel("stat", "Model version served", [("qp_model_version", "")], 0, 0, 4, 5, desc="Registry version behind the production alias"),
    panel("stat", "Live AUC", [("qp_rolling_auc", "")], 4, 0, 4, 5, unit="none", minv=0.5, maxv=1,
          thresholds=[{"color": red, "value": None}, {"color": amber, "value": 0.70}, {"color": green, "value": 0.73}],
          desc="AUC over the last N predictions whose true answer has arrived"),
    panel("stat", "Drift detected", [("qp_drift_detected", "")], 8, 0, 4, 5, minv=0, maxv=1,
          thresholds=[{"color": green, "value": None}, {"color": red, "value": 1}]),
    panel("stat", "Requests / s", [('sum(rate(qp_http_requests_total[1m]))', "")], 12, 0, 4, 5, unit="reqps"),
    panel("stat", "Predictions served", [('sum(qp_predictions_total)', "")], 16, 0, 4, 5),
    panel("stat", "Labels received", [('sum(qp_outcomes_total)', "")], 20, 0, 4, 5),

    panel("timeseries", "Request rate by endpoint", [('sum by (endpoint) (rate(qp_http_requests_total[1m]))', "{{endpoint}}")], 0, 5, 8, 8, unit="reqps"),
    panel("timeseries", "Latency p50 / p95 / p99 (/predict)", [
        ('histogram_quantile(0.5, sum(rate(qp_http_request_seconds_bucket{endpoint="/predict"}[1m])) by (le))', "p50"),
        ('histogram_quantile(0.95, sum(rate(qp_http_request_seconds_bucket{endpoint="/predict"}[1m])) by (le))', "p95"),
        ('histogram_quantile(0.99, sum(rate(qp_http_request_seconds_bucket{endpoint="/predict"}[1m])) by (le))', "p99")], 8, 5, 8, 8, unit="s"),
    panel("timeseries", "Errors (5xx) per second", [('sum(rate(qp_http_requests_total{status=~"5.."}[1m])) or vector(0)', "5xx"),
                                                    ('sum(rate(qp_errors_total[1m])) or vector(0)', "app errors")], 16, 5, 8, 8, unit="reqps"),

    panel("timeseries", "Prediction distribution (median / p10 / p90 of p_correct)", [
        ('histogram_quantile(0.5, sum(rate(qp_prediction_value_bucket[2m])) by (le))', "median"),
        ('histogram_quantile(0.1, sum(rate(qp_prediction_value_bucket[2m])) by (le))', "p10"),
        ('histogram_quantile(0.9, sum(rate(qp_prediction_value_bucket[2m])) by (le))', "p90")], 0, 13, 8, 8, unit="none", minv=0, maxv=1,
          desc="Prediction drift shows up as a shift of this distribution"),
    panel("timeseries", "Rolling AUC / accuracy (live, labelled)", [("qp_rolling_auc", "AUC"), ("qp_rolling_accuracy", "accuracy")],
          8, 13, 8, 8, unit="none", minv=0.5, maxv=1),
    panel("timeseries", "Feature drift (PSI, threshold 0.2)", [("qp_feature_psi", "{{feature}}"), ("qp_prediction_psi", "prediction")],
          16, 13, 8, 8, unit="none", thresholds=[{"color": green, "value": None}, {"color": red, "value": 0.2}]),

    panel("timeseries", "Predictions by model version", [('sum by (model_version) (rate(qp_predictions_total[1m]))', "v{{model_version}}")], 0, 21, 8, 7, unit="ops"),
    panel("timeseries", "Labelled outcomes (correct vs wrong)", [('sum by (correct) (rate(qp_outcomes_total[1m]))', "correct={{correct}}")], 8, 21, 8, 7, unit="ops"),
    panel("timeseries", "Model reloads & retrains", [("qp_model_reloads_total", "reloads"), ("sum(qp_retrain_runs_total)", "retrains")], 16, 21, 8, 7),
]

dash = {"uid": "quizpulse", "title": "QuizPulse - ML service monitoring", "tags": ["quizpulse", "mlops"],
        "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"}, "panels": panels, "templating": {"list": []}}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "docker" / "grafana" / "dashboards" / "quizpulse.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dash, indent=2))
    print("wrote", out, f"({len(panels)} panels)")
