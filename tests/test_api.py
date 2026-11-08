"""API contract tests (FastAPI TestClient + fake model + temp SQLite)."""

from src.serving.model_loader import holder


def _any_student_question(client):
    sid = client.get("/students?limit=1").json()[0]["student_id"]
    skill = client.get("/skills").json()[0]
    q = client.get("/questions", params={"skill": skill}).json()[0]
    return sid, skill, q


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "1", "database": True}


def test_predict_returns_probability_and_features(client):
    sid, skill, q = _any_student_question(client)
    r = client.post("/predict", json={"student_id": sid, "question_id": q, "skill": skill})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_correct"] <= 1 and body["model_version"] == "1"
    assert set(body["features"]) == {"overall_acc", "skill_acc", "skill_attempts", "recent_acc5",
                                     "hours_since_last", "question_difficulty"}


def test_predict_validates_input(client):
    assert client.post("/predict", json={"student_id": "", "question_id": "q", "skill": "s"}).status_code == 422
    assert client.post("/predict", json={"question_id": "q"}).status_code == 422


def test_predict_new_student_cold_start(client):
    sid, skill, q = _any_student_question(client)
    r = client.post("/predict", json={"student_id": "BRAND_NEW", "question_id": q, "skill": skill})
    assert r.status_code == 200 and r.json()["features"]["skill_attempts"] == 0


def test_outcome_roundtrip_adds_attempt(client):
    sid, skill, q = _any_student_question(client)
    pid = client.post("/predict", json={"student_id": sid, "question_id": q, "skill": skill}).json()["prediction_id"]
    r = client.post("/outcomes", json={"prediction_id": pid, "correct": 1})
    assert r.status_code == 200 and r.json()["status"] == "recorded"


def test_outcome_unknown_prediction_is_404(client):
    assert client.post("/outcomes", json={"prediction_id": 99999999, "correct": 1}).status_code == 404


def test_outcome_requires_valid_label_and_identity(client):
    assert client.post("/outcomes", json={"prediction_id": 1, "correct": 5}).status_code == 422
    assert client.post("/outcomes", json={"correct": 1}).status_code == 422


def test_outcome_without_prediction_id_but_full_identity(client):
    sid, skill, q = _any_student_question(client)
    r = client.post("/outcomes", json={"student_id": sid, "question_id": q, "skill": skill, "correct": 0})
    assert r.status_code == 200


def test_weak_topics_sorted_ascending(client):
    sid, _, _ = _any_student_question(client)
    r = client.get(f"/students/{sid}/mastery")
    assert r.status_code == 200
    m = [t["mastery"] for t in r.json()["topics"]]
    assert m == sorted(m) and len(m) >= 3
    weak = client.get(f"/students/{sid}/weak-topics").json()["topics"]
    assert all(t["mastery"] < 0.5 and t["weak"] for t in weak)


def test_next_questions(client):
    sid, _, _ = _any_student_question(client)
    r = client.post("/next-questions", json={"student_id": sid, "n": 4})
    assert r.status_code == 200
    qs = r.json()["questions"]
    assert 1 <= len(qs) <= 4 and all(0 <= q["p_correct"] <= 1 for q in qs)
    assert client.post("/next-questions", json={"student_id": sid, "n": 0}).status_code == 422


def test_503_when_no_model_loaded(client):
    old_model, old_ver = holder.model, holder.version
    holder.model = None
    try:
        r = client.post("/predict", json={"student_id": "a", "question_id": "q", "skill": "s"})
        assert r.status_code == 503
        assert client.get("/health").json()["status"] == "degraded"
    finally:
        holder.model, holder.version = old_model, old_ver


def test_metrics_endpoint_exposes_business_metrics(client):
    sid, skill, q = _any_student_question(client)
    client.post("/predict", json={"student_id": sid, "question_id": q, "skill": skill})
    text = client.get("/metrics").text
    for name in ["qp_http_requests_total", "qp_http_request_seconds_bucket", "qp_predictions_total",
                 "qp_prediction_value_bucket", "qp_model_version", "qp_outcomes_total"]:
        assert name in text, name


def test_pipeline_endpoints(client):
    import time

    from src.common.db import log_run
    log_run("drift", "success", time.time(), details={"status": "ok", "drift_detected": False})
    runs = client.get("/pipeline/runs?kind=drift").json()
    assert runs and runs[0]["kind"] == "drift"
    assert "drift" in client.get("/pipeline/status").json()
    assert "counts" in client.get("/monitoring/summary").json()
