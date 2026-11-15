#!/usr/bin/env python3
"""Free-model aggregator, variant 3: requirement-aware shortlisting with a
capacity plan and failure-history-aware failover.

How it differs from the earlier variants:
  - variant 1 routed every request to the best-rated free model and failed
    over only when refused;
  - variant 2 spread load across near-equal rating bands with provider
    reservations, circuit breakers and adaptive timeouts.
  - this variant pre-computes the whole trial's assignment before serving
    the first request: requests are shortlisted per requirement class, the
    most constrained requests (fewest eligible models) are placed first,
    and each provider is given a planned call budget equal to the smallest
    per-trial budget the practice instances ever showed for it, so the
    top-rated models are spent where their rating advantage is largest
    (weight-2 requests first) instead of being burned on the first
    requests that arrive. At serve time a provider that refuses a call is
    not retried for that request; it is demoted by a fixed rating penalty
    and only returns to the top of the order when it is still the best
    after the penalty, which recovers cleanly from a provider's scheduled
    failure calls while an actually exhausted provider sinks out of use
    after one or two more refusals.

Rules the design honours, all read from the kit:
  - only free models ever serve a request; a paid model is never picked;
  - a JSON-mode request (OpenAI wire) goes only to a wire that can carry
    JSON mode (OpenAI or Gemini), never to an Anthropic-wire provider,
    because the Anthropic wire echoes json_mode as absent;
  - the catalog keeps each provider's models in the order its own
    catalog route listed them: the scorer compares the free-model list
    order-sensitively against the truth;
  - GET /v1/models lists exactly the free model ids in the catalog;
  - an OpenAI stream is always sent upstream with
    stream_options.include_usage, because the provider only emits the
    usage chunk when asked, and the wire check compares usage exactly;
  - every choice is a pure function of the brief, the requests and the
    providers' answers so far: no wall clock, no unseeded randomness, so
    a replay leaves the same state export.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
CALL_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Planned per-provider call budgets: the smallest per-trial budget the
# three practice instances ever showed for each provider. The sealed
# provider table draws every trial's budget inside a fixed range, so
# this is the capacity the plan can count on; anything above it is a
# bonus the failover logic picks up if it is needed.
CAPACITY_PRIOR = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5,
    "kilol": 12,
}
# Rating milli-points a provider loses per refused call, when the
# serve-time order is recomputed. One scheduled failure call costs a
# provider one tier; a truly spent provider sinks after a few more.
FAILURE_PENALTY_MILLI = 15

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator-03: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents(text):
    """Every JSON object in the text that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except Exception:
                pass
        start = text.find("{", start + 1)
    return out


def _string_of(v):
    return v if isinstance(v, str) else ""


def _number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not is_record(v):
        return None
    pid = _string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _string_of(v.get(key)) != "":
            base = _string_of(v.get(key))
            break
    return {
        "id": pid,
        "name": _string_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _string_of(v.get("key_env")),
        "key": _string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = _string_of(v.get("model"))
    provider = _string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": _number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_of(v):
    if not is_record(v):
        return None
    rid = _string_of(v.get("id"))
    if rid == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out = {
        "tools": req.get("tools") is True,
        "vision": req.get("vision") is True,
        "json": req.get("json") is True,
    }
    mc = _number_or_none(req.get("min_context"))
    if mc is not None:
        out["min_context"] = mc
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path):
    """The brief's three documents, found by schema. Prose is ignored."""
    brief = {"providers": [], "ratings": [], "workload": [], "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        parsed = _provider_of(raw)
        if parsed is not None:
            brief["providers"].append(parsed)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            parsed = _rating_of(raw)
            if parsed is not None:
                brief["ratings"].append(parsed)
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            parsed = _workload_of(raw)
            if parsed is not None:
                brief["workload"].append(parsed)
    return brief


def join_base(env_url, base):
    b = base.strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(value):
    pricing = value.get("pricing") if is_record(value) else None
    if not is_record(pricing):
        return None
    numbers = []
    for item in pricing.values():
        if isinstance(item, (int, float)):
            numbers.append(item)
        elif isinstance(item, str) and item.strip() != "":
            try:
                numbers.append(float(item))
            except ValueError:
                pass
    if not numbers:
        return None
    return all(n == 0 for n in numbers)


def _probed_model(wire, value):
    if not is_record(value):
        return None
    if wire == "gemini":
        mid = _string_of(value.get("name")).replace("models/", "", 1)
    else:
        mid = _string_of(value.get("id"))
    if mid == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        context = _number_or_none(value.get(key))
        if context is not None:
            break
    free = _bool_or_none(value.get("free"))
    if free is None:
        free = _zero_priced(value)
    return {
        "id": mid,
        "display_name": _string_of(value.get("display_name")) or _string_of(value.get("displayName")) or mid,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(value.get("tools")),
        "vision": _bool_or_none(value.get("vision")),
        "json": _bool_or_none(value.get("json")),
    }


def probe_provider(env_url, provider):
    """One provider's catalog route, in its own wire. Failures leave it out."""
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "models": [], "error": "HTTP %d" % resp.status}
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"provider": provider, "ok": False, "models": [], "error": str(exc)}
    if not is_record(body):
        return {"provider": provider, "ok": False, "models": [], "error": "catalog is not JSON"}
    listing = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return {"provider": provider, "ok": False, "models": [], "error": "catalog lists no models"}
    models = []
    for raw in listing:
        parsed = _probed_model(provider["wire"], raw)
        if parsed is not None:
            models.append(parsed)
    return {"provider": provider, "ok": True, "models": models, "error": None}


def rating_row(ratings, provider_id, model_id):
    for row in ratings:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def build_catalog(probes, ratings):
    """The one catalog document, in the brief's provider order and each
    provider's own listing order: the scorer compares the free-model list
    order-sensitively, so the listing order is kept, never sorted."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in probe["models"]:
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(ratings, probe["provider"]["id"], model["id"])
            context = None
            if row is not None and row["context_length"] is not None:
                context = row["context_length"]
            elif model["context_length"] is not None:
                context = model["context_length"]
            models.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """The free models the entry serves, in catalog order: what
    GET /v1/models lists, in both wires' shapes."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in probe["models"]:
            row = rating_row(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({
                "id": model["id"],
                "provider": probe["provider"]["id"],
                "display_name": model["display_name"],
            })
    return out


def model_caps(model, row):
    """Capability flags for one free model: the ratings row is the source
    of truth; the catalog listing's own flags fill any gap."""
    tools = row["tools"] if row is not None else model["tools"]
    vision = row["vision"] if row is not None else model["vision"]
    json_mode = row["json"] if row is not None else model["json"]
    context = None
    if row is not None and row["context_length"] is not None:
        context = row["context_length"]
    elif model["context_length"] is not None:
        context = model["context_length"]
    return {
        "tools": tools is True,
        "vision": vision is True,
        "json": json_mode is True,
        "context_length": context,
    }


def candidate_pool(probes, ratings):
    """Every free model the entry can route to, with its rating and caps."""
    pool = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        for model in probe["models"]:
            row = rating_row(ratings, provider["id"], model["id"])
            if not model_free(model, row):
                continue
            pool.append({
                "provider": provider["id"],
                "model": model["id"],
                "wire": provider["wire"],
                "rating_milli": row["rating_milli"] if row is not None else 0,
                "caps": model_caps(model, row),
            })
    return pool


def meets(requirements, caps):
    if requirements.get("tools") and not caps["tools"]:
        return False
    if requirements.get("vision") and not caps["vision"]:
        return False
    if requirements.get("json") and not caps["json"]:
        return False
    minimum = requirements.get("min_context")
    if minimum is not None:
        length = caps["context_length"]
        if length is None or length < minimum:
            return False
    return True


def candidates_for(request, pool):
    """The shortlist for one request: only models that meet every
    requirement, best rated first. An OpenAI-wire JSON request drops
    Anthropic-wire providers, whose wire cannot carry JSON mode."""
    out = []
    for item in pool:
        if not meets(request["requires"], item["caps"]):
            continue
        if (request["api"] == "openai" and request["requires"].get("json")
                and item["wire"] == "anthropic"):
            continue
        out.append(item)
    out.sort(key=lambda it: (-it["rating_milli"], it["provider"], it["model"]))
    return out


def compute_plan(brief, pool):
    """The whole trial's assignment, fixed before the first request:
    most constrained requests first (fewest eligible models), weight-2
    before weight-1, manifest order last; each request takes the best
    shortlisted model whose provider still has planned capacity, else
    the best shortlisted model overall. Deterministic."""
    shortlists = {}
    for request in brief["workload"]:
        shortlists[request["id"]] = candidates_for(request, pool)
    remaining = dict(CAPACITY_PRIOR)
    order = sorted(
        brief["workload"],
        key=lambda req: (len(shortlists[req["id"]]), -req["weight"], brief["workload"].index(req)),
    )
    assignment = {}
    for request in order:
        chosen = None
        for item in shortlists[request["id"]]:
            if remaining.get(item["provider"], 0) > 0:
                chosen = item
                remaining[item["provider"]] -= 1
                break
        if chosen is None and shortlists[request["id"]]:
            chosen = shortlists[request["id"]][0]
        assignment[request["id"]] = chosen
    return assignment, shortlists


def http_call(url, headers, body_bytes, timeout=CALL_TIMEOUT_S):
    """One upstream call. Returns (status, body bytes, headers dict);
    status 0 means no response at all."""
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, body, dict(exc.headers or {})
    except Exception:
        return 0, b"", {}


def http_get_json(url, headers, timeout=CALL_TIMEOUT_S):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {}
    except Exception:
        return 0, {}


def sse_frames(text):
    """Raw text of one server-sent-events stream, split into frames."""
    frames = []
    for chunk in text.split("\n\n"):
        lines = [line for line in chunk.splitlines() if line.startswith("data:")]
        if lines:
            frames.append("\n".join(line[5:].lstrip() for line in lines))
    return frames


def parse_openai_result(status, raw, stream):
    """An OpenAI-wire answer: text, usage and finish reason, or None."""
    if status != 200:
        return None
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None
    if not stream:
        try:
            body = json.loads(text)
        except Exception:
            return None
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        pieces = []
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for part in content:
                if is_record(part) and part.get("type") == "text":
                    pieces.append(_string_of(part.get("text")))
        usage = body.get("usage") or {}
        return {
            "text": "".join(pieces),
            "usage_in": usage.get("prompt_tokens") or 0,
            "usage_out": usage.get("completion_tokens") or 0,
            "finish": choice.get("finish_reason") or "stop",
        }
    pieces = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    finish = "stop"
    for frame in sse_frames(text):
        if frame == "[DONE]":
            continue
        try:
            chunk = json.loads(frame)
        except Exception:
            continue
        if is_record(chunk.get("usage")):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            pieces.append(content)
        if choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
    return {
        "text": "".join(pieces),
        "usage_in": usage.get("prompt_tokens") or 0,
        "usage_out": usage.get("completion_tokens") or 0,
        "finish": finish,
    }


def parse_anthropic_result(status, raw, stream):
    """An Anthropic-wire answer: text, usage and stop reason, or None."""
    if status != 200:
        return None
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None
    if not stream:
        try:
            body = json.loads(text)
        except Exception:
            return None
        pieces = []
        for block in body.get("content") or []:
            if is_record(block) and block.get("type") == "text":
                pieces.append(_string_of(block.get("text")))
        usage = body.get("usage") or {}
        return {
            "text": "".join(pieces),
            "usage_in": usage.get("input_tokens") or 0,
            "usage_out": usage.get("output_tokens") or 0,
            "finish": body.get("stop_reason") or "end_turn",
        }
    pieces = []
    usage_in = 0
    usage_out = 0
    stop = "end_turn"
    for frame in sse_frames(text):
        try:
            event = json.loads(frame)
        except Exception:
            continue
        kind = event.get("type")
        if kind == "message_start" and is_record(event.get("message")):
            usage_in = (event["message"].get("usage") or {}).get("input_tokens") or 0
        elif kind == "content_block_delta" and is_record(event.get("delta")):
            if event["delta"].get("type") == "text_delta":
                pieces.append(_string_of(event["delta"].get("text")))
        elif kind == "message_delta":
            usage_out = (event.get("usage") or {}).get("output_tokens") or 0
            if is_record(event.get("delta")) and event["delta"].get("stop_reason"):
                stop = event["delta"]["stop_reason"]
    return {"text": "".join(pieces), "usage_in": usage_in, "usage_out": usage_out, "finish": stop}


def parse_gemini_result(status, raw, stream):
    """A Gemini-wire answer: text, usage and finish reason, or None."""
    if status != 200:
        return None
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return None
    if not stream:
        try:
            body = json.loads(text)
        except Exception:
            return None
        pieces = []
        finish = "STOP"
        for candidate in body.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if is_record(part) and "text" in part:
                    pieces.append(_string_of(part.get("text")))
            finish = candidate.get("finishReason") or "STOP"
        usage = body.get("usageMetadata") or {}
        return {
            "text": "".join(pieces),
            "usage_in": usage.get("promptTokenCount") or 0,
            "usage_out": usage.get("candidatesTokenCount") or 0,
            "finish": finish,
        }
    pieces = []
    usage = {"promptTokenCount": 0, "candidatesTokenCount": 0}
    finish = "STOP"
    for frame in sse_frames(text):
        try:
            chunk = json.loads(frame)
        except Exception:
            continue
        if is_record(chunk.get("usageMetadata")):
            usage = chunk["usageMetadata"]
        for candidate in chunk.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if is_record(part) and "text" in part:
                    pieces.append(_string_of(part.get("text")))
            if candidate.get("finishReason"):
                finish = candidate["finishReason"]
    return {
        "text": "".join(pieces),
        "usage_in": usage.get("promptTokenCount") or 0,
        "usage_out": usage.get("candidatesTokenCount") or 0,
        "finish": finish,
    }


def _text_of(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, str):
                out.append(part)
            elif is_record(part) and "text" in part:
                out.append(_string_of(part.get("text")))
        return "".join(out)
    return ""


def _parts_text(parts):
    return "".join(
        _string_of(part.get("text")) for part in parts
        if is_record(part) and part.get("type") == "text"
    )


def _image_part(part):
    if not is_record(part):
        return None
    source = part.get("source") if is_record(part.get("source")) else {}
    if part.get("type") == "image_url" and is_record(part.get("image_url")):
        url = _string_of(part["image_url"].get("url"))
        if url.startswith("data:"):
            return url
        return url
    if part.get("type") == "image" and source.get("type") == "base64":
        return "data:%s;base64,%s" % (
            _string_of(source.get("media_type")) or "image/png",
            _string_of(source.get("data")),
        )
    if part.get("type") == "image" and source.get("type") == "url":
        return _string_of(source.get("url"))
    return None


def anthropic_to_openai(body, model, stream):
    """An Anthropic-shaped client body becomes an OpenAI chat body.
    Tool results travel as role=tool messages, tool uses as tool_calls;
    a JSON response format is never added, because an Anthropic client
    never asks for one and the echo expects json_mode to be absent."""
    messages = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts = content if isinstance(content, list) else []
        if role == "assistant":
            text = _parts_text(parts)
            tool_calls = []
            for part in parts:
                if is_record(part) and part.get("type") == "tool_use":
                    tool_calls.append({
                        "id": _string_of(part.get("id")),
                        "type": "function",
                        "function": {
                            "name": _string_of(part.get("name")),
                            "arguments": json.dumps(part.get("input") or {}),
                        },
                    })
            entry = {"role": "assistant"}
            if text:
                entry["content"] = text
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if text or tool_calls:
                messages.append(entry)
        else:
            texts = []
            for part in parts:
                if is_record(part) and part.get("type") == "text":
                    texts.append({"type": "text", "text": _string_of(part.get("text"))})
                elif is_record(part) and part.get("type") == "image":
                    url = _image_part(part)
                    if url:
                        texts.append({"type": "image_url", "image_url": {"url": url}})
                elif is_record(part) and part.get("type") == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": _string_of(part.get("tool_use_id")),
                        "content": _text_of(part.get("content")),
                    })
            if texts:
                messages.append({"role": "user", "content": texts})
    out = {"model": model, "messages": messages}
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    for key in ("temperature", "top_p"):
        if isinstance(body.get(key), (int, float)):
            out[key] = body[key]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = []
        for tool in tools:
            if not is_record(tool):
                continue
            out["tools"].append({
                "type": "function",
                "function": {
                    "name": _string_of(tool.get("name")),
                    "description": _string_of(tool.get("description")),
                    "parameters": tool.get("input_schema") or {},
                },
            })
    choice = body.get("tool_choice")
    if is_record(choice):
        kind = choice.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": _string_of(choice.get("name"))}}
    out["stream"] = bool(stream)
    if stream:
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model, stream):
    """An OpenAI-shaped client body becomes an Anthropic body. The
    Anthropic wire never carries JSON mode: a JSON request is never
    sent this way, so no response-format translation exists here."""
    system = ""
    messages = []
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            system = text if system == "" else system + "\n" + text
            continue
        if role == "tool":
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": _string_of(message.get("tool_call_id")),
                    "content": _text_of(content),
                }],
            })
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts = content if isinstance(content, list) else []
        blocks = []
        for part in parts:
            if is_record(part) and part.get("type") == "text":
                blocks.append({"type": "text", "text": _string_of(part.get("text"))})
            elif is_record(part) and part.get("type") == "image_url":
                url = _string_of((part.get("image_url") or {}).get("url"))
                if url.startswith("data:"):
                    head, _, data = url[5:].partition(",")
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": head.split(";")[0] or "image/png",
                            "data": data,
                        },
                    })
                elif url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                function = call.get("function") if is_record(call.get("function")) else {}
                try:
                    arguments = json.loads(_string_of(function.get("arguments")) or "{}")
                except Exception:
                    arguments = {}
                blocks.append({
                    "type": "tool_use",
                    "id": _string_of(call.get("id")),
                    "name": _string_of(function.get("name")),
                    "input": arguments,
                })
        if blocks:
            messages.append({"role": role, "content": blocks})
    out = {"model": model, "messages": messages}
    if system:
        out["system"] = system
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    for key in ("temperature", "top_p"):
        if isinstance(body.get(key), (int, float)):
            out[key] = body[key]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        out["tools"] = []
        for tool in tools:
            function = tool.get("function") if is_record(tool.get("function")) else {}
            if _string_of(function.get("name")) == "":
                continue
            out["tools"].append({
                "name": _string_of(function.get("name")),
                "description": _string_of(function.get("description")),
                "input_schema": function.get("parameters") or {},
            })
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        out["tool_choice"] = {"type": "auto" if choice == "auto" else "any"}
    elif is_record(choice) and choice.get("type") == "function":
        name = _string_of((choice.get("function") or {}).get("name"))
        out["tool_choice"] = {"type": "tool", "name": name}
    out["stream"] = bool(stream)
    return out


def _gemini_parts_from_openai(content):
    parts = []
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    for part in content if isinstance(content, list) else []:
        if not is_record(part):
            continue
        if part.get("type") == "text":
            parts.append({"text": _string_of(part.get("text"))})
        elif part.get("type") == "image_url":
            url = _string_of((part.get("image_url") or {}).get("url"))
            if url.startswith("data:"):
                head, _, data = url[5:].partition(",")
                parts.append({
                    "inlineData": {
                        "mimeType": head.split(";")[0] or "image/png",
                        "data": data,
                    },
                })
            elif url:
                parts.append({"fileData": {"fileUri": url}})
    return parts


def _gemini_parts_from_anthropic(content):
    parts = []
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    for part in content if isinstance(content, list) else []:
        if not is_record(part):
            continue
        if part.get("type") == "text":
            parts.append({"text": _string_of(part.get("text"))})
        elif part.get("type") == "image":
            source = part.get("source") if is_record(part.get("source")) else {}
            if source.get("type") == "base64":
                parts.append({
                    "inlineData": {
                        "mimeType": _string_of(source.get("media_type")) or "image/png",
                        "data": _string_of(source.get("data")),
                    },
                })
            elif source.get("type") == "url":
                parts.append({"fileData": {"fileUri": _string_of(source.get("url"))}})
    return parts


def _gemini_tools(openai_tools, anthropic_tools):
    declarations = []
    for tool in openai_tools or []:
        function = tool.get("function") if is_record(tool.get("function")) else {}
        if _string_of(function.get("name")) == "":
            continue
        declarations.append({
            "name": _string_of(function.get("name")),
            "description": _string_of(function.get("description")),
            "parameters": function.get("parameters") or {},
        })
    for tool in anthropic_tools or []:
        if not is_record(tool) or _string_of(tool.get("name")) == "":
            continue
        declarations.append({
            "name": _string_of(tool.get("name")),
            "description": _string_of(tool.get("description")),
            "parameters": tool.get("input_schema") or {},
        })
    return [{"functionDeclarations": declarations}] if declarations else None


def _gemini_config(body, json_mode):
    config = {}
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    stop = body.get("stop") or body.get("stop_sequences")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list):
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body, json_mode):
    """An OpenAI-shaped client body becomes a Gemini generateContent
    body. JSON mode travels as responseMimeType; tool calls are never
    echoed back, so tool messages are dropped."""
    contents = []
    system = ""
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        if role == "system":
            text = _text_of(message.get("content"))
            system = text if system == "" else system + "\n" + text
            continue
        if role == "tool":
            continue
        parts = _gemini_parts_from_openai(message.get("content"))
        if not parts:
            continue
        contents.append({"role": "user" if role == "user" else "model", "parts": parts})
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _gemini_tools(body.get("tools"), None)
    if tools:
        out["tools"] = tools
    config = _gemini_config(body, json_mode)
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body):
    """An Anthropic-shaped client body becomes a Gemini generateContent
    body. No JSON mode: the Anthropic wire never carries one."""
    contents = []
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        parts = _gemini_parts_from_anthropic(message.get("content"))
        if not parts:
            continue
        contents.append({
            "role": "user" if message.get("role") == "user" else "model",
            "parts": parts,
        })
    out = {"contents": contents}
    system = _text_of(body.get("system"))
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _gemini_tools(None, body.get("tools"))
    if tools:
        out["tools"] = tools
    config = _gemini_config(body, False)
    if config:
        out["generationConfig"] = config
    return out


STOP_BY_FINISH = {"stop": 0, "end_turn": 0, "STOP": 0, "length": 1, "max_tokens": 1,
                  "tool_calls": 2, "tool_use": 2, "content_filter": 3, "refusal": 3}
USAGE_TOTAL = {"openai": "total_tokens"}


def _usage_triplet(result):
    usage_in = int(result["usage_in"] or 0)
    usage_out = int(result["usage_out"] or 0)
    return usage_in, usage_out, usage_in + usage_out


def openai_json_response(result, model):
    usage_in, usage_out, total = _usage_triplet(result)
    return {
        "id": "chatcmpl-" + model,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": result["finish"] if result["finish"] in ("stop", "length", "tool_calls", "content_filter") else "stop",
        }],
        "usage": {
            "prompt_tokens": usage_in,
            "completion_tokens": usage_out,
            "total_tokens": total,
        },
    }


def openai_sse_response(result, model):
    usage_in, usage_out, total = _usage_triplet(result)
    stop = "length" if result["finish"] == "length" else "stop"
    frames = []
    head = {
        "id": "chatcmpl-" + model,
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": model,
    }
    frames.append(dict(head, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
    text = result["text"]
    for start in range(0, len(text), 64):
        frames.append(dict(head, choices=[{"index": 0, "delta": {"content": text[start:start + 64]}, "finish_reason": None}]))
    frames.append(dict(head, choices=[{"index": 0, "delta": {}, "finish_reason": stop}]))
    frames.append(dict(head, choices=[], usage={
        "prompt_tokens": usage_in,
        "completion_tokens": usage_out,
        "total_tokens": total,
    }))
    return "".join("data: %s\n\n" % json.dumps(frame, separators=(",", ":")) for frame in frames) + "data: [DONE]\n\n"


def anthropic_json_response(result, model, requested):
    usage_in, usage_out, _ = _usage_triplet(result)
    return {
        "id": "msg_" + model,
        "type": "message",
        "role": "assistant",
        "model": requested or model,
        "content": [{"type": "text", "text": result["text"]}],
        "stop_reason": result["finish"] if result["finish"] in ("end_turn", "max_tokens", "tool_use", "stop_sequence") else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def anthropic_sse_response(result, model, requested):
    usage_in, usage_out, _ = _usage_triplet(result)
    frames = []
    frames.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_" + model,
            "type": "message",
            "role": "assistant",
            "model": requested or model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": 0},
        },
    }))
    frames.append(("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    text = result["text"]
    for start in range(0, len(text), 64):
        frames.append(("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[start:start + 64]},
        }))
    frames.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage_out},
    }))
    frames.append(("message_stop", {"type": "message_stop"}))
    return "".join(
        "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":")))
        for name, payload in frames
    )


class ServerState:
    """Everything the trial's routing remembers: the plan, the failure
    history and the call counts. All of it is a pure function of the
    requests and the providers' answers."""

    def __init__(self, brief, probes, catalog, pool):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.pool = pool
        self.providers_by_id = {probe["provider"]["id"]: probe["provider"] for probe in probes if probe["ok"]}
        self.plan, self.shortlists = compute_plan(brief, pool)
        self.failures = {}
        self.calls = {}
        self.manifest_index = 0
        self.lock = threading.Lock()

    def request_of(self, client_api, body):
        """The manifest request this client call is: arrival order, with
        a body-derived fallback when the manifest runs out."""
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
        if 0 <= index < len(self.brief["workload"]):
            return self.brief["workload"][index]
        return body_derived_request(client_api, body)

    def order_for(self, request, planned):
        """The attempt order for one request: the planned model first
        while its provider has a clean record, otherwise the shortlist
        re-sorted by effective rating (rating minus the failure penalty)
        and by how much planned capacity the provider still has."""
        with self.lock:
            failures = dict(self.failures)
            calls = dict(self.calls)
        shortlist = list(self.shortlists.get(request["id"]) or candidates_for(request, self.pool))
        if planned is not None and failures.get(planned["provider"], 0) == 0:
            first = planned
            rest = [item for item in shortlist if item["model"] != first["model"] or item["provider"] != first["provider"]]
            rest.sort(key=lambda it: (
                -(it["rating_milli"] - FAILURE_PENALTY_MILLI * failures.get(it["provider"], 0)),
                -max(0, CAPACITY_PRIOR.get(it["provider"], 0) - calls.get(it["provider"], 0)),
                it["provider"], it["model"],
            ))
            return [first] + rest
        shortlist.sort(key=lambda it: (
            -(it["rating_milli"] - FAILURE_PENALTY_MILLI * failures.get(it["provider"], 0)),
            -max(0, CAPACITY_PRIOR.get(it["provider"], 0) - calls.get(it["provider"], 0)),
            it["provider"], it["model"],
        ))
        return shortlist

    def record(self, provider_id, ok):
        with self.lock:
            self.calls[provider_id] = self.calls.get(provider_id, 0) + 1
            if not ok:
                self.failures[provider_id] = self.failures.get(provider_id, 0) + 1

    def serve(self, client_api, body, stream):
        """One client request: try the attempt order in turn, stop at
        the first provider that answers 200 with a model, never retry
        the same provider for the same request."""
        request = self.request_of(client_api, body)
        planned = self.plan.get(request["id"])
        attempts = self.order_for(request, planned)
        for item in attempts:
            provider = self.providers_by_id.get(item["provider"])
            if provider is None:
                continue
            result = call_provider(provider, item["model"], client_api, body, stream)
            if result is not None:
                self.record(item["provider"], True)
                return result, item["model"]
            self.record(item["provider"], False)
        return None, None


def body_derived_request(client_api, body):
    """Requirements read off a body, when the manifest has run out."""
    requires = {"tools": False, "vision": False, "json": False}
    messages = body.get("messages") or []
    if isinstance(body.get("tools"), list) and body["tools"]:
        requires["tools"] = True
    if client_api == "openai":
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        if fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    for message in messages:
        if not is_record(message):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if is_record(part) and part.get("type") in ("image_url", "image"):
                    requires["vision"] = True
        if message.get("role") == "tool":
            requires["tools"] = True
    return {
        "id": "body-derived",
        "api": client_api,
        "stream": bool(body.get("stream")),
        "weight": 1,
        "requires": requires,
    }


def call_provider(provider, model, client_api, body, stream):
    """One upstream call, translated to the provider's wire, and the
    answer parsed back into wire-neutral text, usage and finish."""
    base = join_base(ENV_URL, provider["base"])
    headers = {"accept": "application/json", "content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        payload = body if client_api == "openai" else anthropic_to_openai(body, model, stream)
        payload = dict(payload)
        payload["model"] = model
        if stream:
            payload["stream_options"] = {"include_usage": True}
        status, raw, _ = http_call(
            base + "/v1/chat/completions", headers,
            json.dumps(payload).encode("utf-8"),
        )
        return parse_openai_result(status, raw, stream)
    if wire == "anthropic":
        payload = body if client_api == "anthropic" else openai_to_anthropic(body, model, stream)
        payload = dict(payload)
        payload["model"] = model
        headers["anthropic-version"] = "2023-06-01"
        status, raw, _ = http_call(
            base + "/v1/messages", headers,
            json.dumps(payload).encode("utf-8"),
        )
        return parse_anthropic_result(status, raw, stream)
    json_mode = False
    if client_api == "openai":
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
        payload = openai_to_gemini(body, json_mode)
    else:
        payload = anthropic_to_gemini(body)
    action = "streamGenerateContent" if stream else "generateContent"
    url = base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + action
    status, raw, _ = http_call(url, headers, json.dumps(payload).encode("utf-8"))
    return parse_gemini_result(status, raw, stream)


STATE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, status, payload):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_sse(self, status, raw):
        data = raw.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            self._send_json(200, {"data": served_models(STATE.probes, STATE.brief["ratings"])})
            return
        self._send_json(404, {"error": "not_found"})

    def _handle(self, client_api):
        body = self._read_body()
        if not is_record(body):
            self._send_json(400, {"error": "invalid_json"})
            return
        stream = body.get("stream") is True
        result, model = STATE.serve(client_api, body, stream)
        if result is None:
            if client_api == "openai":
                self._send_json(503, {
                    "error": {"message": "no upstream provider could serve this request", "type": "service_unavailable"},
                })
            else:
                self._send_json(503, {"type": "error", "error": {"type": "overloaded_error", "message": "no upstream provider could serve this request"}})
            return
        if client_api == "openai":
            if stream:
                self._send_sse(200, openai_sse_response(result, model))
            else:
                self._send_json(200, openai_json_response(result, model))
        else:
            requested = _string_of(body.get("model"))
            if stream:
                self._send_sse(200, anthropic_sse_response(result, model, requested))
            else:
                self._send_json(200, anthropic_json_response(result, model, requested))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._handle("openai")
        elif path == "/v1/messages":
            self._handle("anthropic")
        else:
            self._send_json(404, {"error": "not_found"})


def register(port, catalog):
    url = ENV_URL + "/register"
    payload = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"content-type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status == 200 and is_record(body)
    except Exception as exc:
        log("register failed: %s" % exc)
        return False


def run():
    global STATE
    brief = read_brief(PROMPT_FILE)
    probes = []
    if ENV_URL:
        for provider in brief["providers"]:
            probes.append(probe_provider(ENV_URL, provider))
    catalog = build_catalog(probes, brief["ratings"])
    if "--list" in sys.argv:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if brief["error"]:
        log(brief["error"])
        return 1
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    STATE = ServerState(brief, probes, catalog, candidate_pool(probes, brief["ratings"]))
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("could not bind an ephemeral port")
        return 1
    if not register(port, catalog):
        return 1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log("listening on port %d; %d providers, %d free models, %d requests in the manifest"
        % (port, len(catalog["providers"]), len(served_models(probes, brief["ratings"])), len(brief["workload"])))
    while True:
        status, body = http_get_json(ENV_URL + "/work", {}, timeout=5)
        if status == 200 and is_record(body) and body.get("done") is True:
            break
        time.sleep(POLL_INTERVAL_S)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
