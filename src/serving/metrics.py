"""Prometheus instrumentation of the API (Lecture 9: white-box monitoring, labels > hierarchy)."""
from prometheus_client import Counter, Gauge, Histogram, Info

HTTP_REQUESTS = Counter("qp_http_requests_total", "HTTP requests", ["method", "endpoint", "status"])
HTTP_LATENCY = Histogram("qp_http_request_seconds", "HTTP request latency", ["endpoint"],
                         buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
PREDICTIONS = Counter("qp_predictions_total", "Predictions served", ["model_version"])
PREDICTION_VALUE = Histogram("qp_prediction_value", "Distribution of predicted p_correct",
                             buckets=(.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0))
OUTCOMES = Counter("qp_outcomes_total", "Delayed labels received", ["correct"])
ERRORS = Counter("qp_errors_total", "Application errors", ["endpoint", "type"])
MODEL_VERSION = Gauge("qp_model_version", "Registry version currently served")
MODEL_RELOADS = Counter("qp_model_reloads_total", "Model (re)loads from the registry")
MODEL_LOADED = Gauge("qp_model_loaded", "1 if a model is loaded")
FEATURE_LATENCY = Histogram("qp_feature_build_seconds", "Online feature construction time",
                            buckets=(.001, .005, .01, .025, .05, .1, .25, .5))
BUILD_INFO = Info("qp_build", "QuizPulse API build information")
