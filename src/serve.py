"""Serve a trained domain adapter behind the decision-API contract.

    python src/serve.py --adapter models/banking
    curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d @req.json

Endpoints:
    POST /v1/systemone   typed questions over a shared state
    GET  /health         readiness
    GET  /               a page for trying it by hand

Standard library only apart from the model stack. The model loads once and
every request is a single forward pass over a three-level attention tree
(state -> question branches -> option sub-branches), so Q questions cost one
pass rather than Q.
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><meta charset=utf-8>
<title>domain decision model</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;
      background:#fbfbfd;color:#1d1d1f}
 h1{font-size:1.3rem;margin:0 0 .2rem} .sub{color:#6e6e73;margin:0 0 1.4rem}
 textarea{width:100%;font:13px/1.45 ui-monospace,monospace;padding:.7rem;
          border:1px solid #d2d2d7;border-radius:8px;background:#fff}
 button{font:inherit;padding:.5rem 1.1rem;border:0;border-radius:8px;
        background:#1d1d1f;color:#fff;cursor:pointer;margin:.8rem 0}
 button:disabled{opacity:.5;cursor:default}
 pre{background:#fff;border:1px solid #d2d2d7;border-radius:8px;padding:.8rem;
     overflow:auto;font-size:13px}
 .t{color:#6e6e73;font-size:13px}
</style>
<h1>domain decision model</h1>
<p class=sub>One forward pass answers every question about the shared state.
Option order and question order cannot affect the answer.</p>
<textarea id=q rows=20></textarea>
<button id=go>Decide</button> <span class=t id=t></span>
<pre id=out>&mdash;</pre>
<script>
const demo = {
  model: "local",
  state: "My card got swallowed by the ATM yesterday. Also, can you tell me when my next payment is due?",
  questions: {
    intent: {type:"choice", instructions:"What does this customer message ask for?",
      criteria:{damaged_card:"reporting a card that is physically damaged",
                bill_due:"asking when a bill or payment is due",
                report_lost_card:"reporting a card lost or stolen",
                out_of_scope:"the message is not a banking or card request at all"}}
  }
};
q.value = JSON.stringify(demo, null, 2);
go.onclick = async () => {
  go.disabled = true; t.textContent = "\\u2026"; const t0 = performance.now();
  try {
    const r = await fetch("/v1/systemone", {method:"POST",
      headers:{"content-type":"application/json"}, body:q.value});
    out.textContent = JSON.stringify(await r.json(), null, 2);
    t.textContent = Math.round(performance.now()-t0) + " ms round trip";
  } catch (e) { out.textContent = String(e); t.textContent = ""; }
  go.disabled = false;
};
</script>"""

STATE = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, json.dumps({"status": "ok",
                                        "adapter": STATE.get("adapter")}))
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode())
        except Exception as e:                                # noqa: BLE001
            self._send(400, json.dumps({"error": f"bad JSON: {e}"}))
            return
        try:
            t0 = time.time()
            out = STATE["decide"](req)
            out["timing_ms"] = round((time.time() - t0) * 1000)
            self._send(200, json.dumps(out))
        except ValueError as e:                  # limits and schema problems
            self._send(422, json.dumps({"error": str(e)}))
        except Exception as e:                                # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}))

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    import decision_api
    from config import BASE
    from pointer import PointerHead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if a.dtype == "fp32" or device == "cpu" \
        else torch.bfloat16
    print(f"loading {BASE} + {a.adapter} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype,
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, a.adapter).to(device).eval()

    hp = os.path.join(a.adapter, "pointer_head.pt")
    if not os.path.exists(hp):
        raise SystemExit(f"no pointer head at {hp}")
    head = PointerHead(model.config.hidden_size).to(device).float()
    head.load_state_dict(torch.load(hp))
    head.eval()

    STATE["adapter"] = a.adapter
    STATE["decide"] = lambda req: decision_api.decide(model, head, tok, req,
                                                      device, dtype)
    print(f"ready on http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
