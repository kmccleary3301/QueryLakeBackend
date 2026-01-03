from QueryLake.runtime.request_context import (
    RequestContext,
    get_request_context,
    get_request_id,
    set_request_context,
    set_request_id,
)


def test_request_context_roundtrip():
    ctx = RequestContext(request_id="req_test", route="/v1/chat/completions")
    set_request_context(ctx)
    assert get_request_context() == ctx
    assert get_request_id() == "req_test"


def test_set_request_id_generates_default():
    req_id = set_request_id(None)
    assert req_id is not None
    assert req_id.startswith("req_")
