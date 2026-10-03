"""
Minimal OTLP/HTTP trace receiver.

vLLM exports one real OpenTelemetry span per request (named "llm_request")
when started with --otlp-traces-endpoint, carrying server-measured
gen_ai.latency.time_in_queue / time_in_model_prefill / time_in_model_decode
-- a genuine phase breakdown from the engine itself, not something we can
reconstruct from the client side. A single process emitting a handful of
spans doesn't need a full OpenTelemetry Collector: decoding the OTLP/HTTP
protobuf payload and handing each span to a callback is the entire job.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)


def _decode_attr_value(value):
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind in ("int_value", "double_value", "bool_value", "string_value"):
        return getattr(value, kind)
    return str(getattr(value, kind))


def _span_to_dict(span, resource_attrs: dict) -> dict:
    attributes = dict(resource_attrs)
    for kv in span.attributes:
        attributes[kv.key] = _decode_attr_value(kv.value)
    return {
        "trace_id": span.trace_id.hex(),
        "span_id": span.span_id.hex(),
        "name": span.name,
        "start_time_unix_nano": span.start_time_unix_nano,
        "end_time_unix_nano": span.end_time_unix_nano,
        "attributes": attributes,
    }


class OtelSpanReceiver:
    """Runs a background HTTP server that accepts OTLP/HTTP trace exports
    and collects the decoded spans in self.spans."""

    def __init__(self, host: str = "127.0.0.1", port: int = 4318) -> None:
        self.spans: list[dict] = []
        self._lock = threading.Lock()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass  # silence per-request access logging

            def do_POST(self) -> None:
                if self.path != "/v1/traces":
                    self.send_response(404)
                    self.end_headers()
                    return

                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                request = ExportTraceServiceRequest()
                request.ParseFromString(body)

                spans = []
                for resource_spans in request.resource_spans:
                    resource_attrs = {
                        kv.key: _decode_attr_value(kv.value)
                        for kv in resource_spans.resource.attributes
                    }
                    for scope_spans in resource_spans.scope_spans:
                        for span in scope_spans.spans:
                            spans.append(_span_to_dict(span, resource_attrs))

                with receiver._lock:
                    receiver.spans.extend(spans)

                response = ExportTraceServiceResponse().SerializeToString()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1/traces"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
