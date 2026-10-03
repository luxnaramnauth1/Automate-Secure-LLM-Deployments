"""GlobalAI - Activity 3: semantic response cache (Redis) + complexity-based model routing.

Run the self-test:  python cost_optimizer.py --prompts ../load_testing/test_prompts.csv
Uses a real Redis if REDIS_URL is reachable, otherwise falls back to fakeredis (in-memory).
"""
import argparse, csv, hashlib, json, math, os, re, time
from collections import Counter as WordCounter
from dataclasses import dataclass

try:
    from prometheus_client import Counter, Gauge
    CACHE_HITS = Counter("llm_cache_hits_total", "Cache hits", ["cache_type"])
    CACHE_MISSES = Counter("llm_cache_misses_total", "Cache misses", ["cache_type"])
    ROUTED = Counter("llm_model_requests_total", "Requests by model", ["model"])
    HIT_RATIO = Gauge("llm_cache_hit_ratio", "Cache hit ratio", ["cache_type"])
except ImportError:  # metrics optional
    class _N:
        def labels(self, **k): return self
        def inc(self, *a): pass
        def set(self, *a): pass
    CACHE_HITS = CACHE_MISSES = ROUTED = HIT_RATIO = _N()

# ---- pricing (USD per 1K tokens, from the starter module) -------------------
MODELS = {
    "llama-3.1-8b":  {"in": 0.0005, "out": 0.0015},
    "llama-3.1-70b": {"in": 0.003,  "out": 0.015},
}

# ---- TTL policy: 24h general, 7 days for classification-style queries -------
TTL_GENERAL = 24 * 3600
TTL_CLASSIFICATION = 7 * 24 * 3600
_CLASSIFY = re.compile(r"\b(rate|rates|routing number|hours|minimum balance|limit|fee|reset|password|beneficiary|what is|what are)\b", re.I)


def ttl_for(prompt: str) -> int:
    return TTL_CLASSIFICATION if _CLASSIFY.search(prompt) and len(prompt.split()) < 12 else TTL_GENERAL


# ---- embedding: hashed word + char-trigram vectors (swap for a real encoder in prod)
DIM = 512
STOP = {"the","a","an","is","are","do","does","i","my","me","can","you","your","to","of","for","on","in","what","how","please"}


def embed(text: str):
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = [w for w in t.split() if w not in STOP]
    feats = WordCounter(words)
    joined = " ".join(words)
    for i in range(len(joined) - 2):
        feats["#" + joined[i:i + 3]] += 0.3
    v = [0.0] * DIM
    for f, c in feats.items():
        v[int(hashlib.md5(f.encode()).hexdigest(), 16) % DIM] += c
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a, b): return sum(x * y for x, y in zip(a, b))


def get_redis():
    import redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        r = redis.from_url(url, socket_connect_timeout=0.5); r.ping(); return r
    except Exception:
        import fakeredis; return fakeredis.FakeRedis()


class SemanticCache:
    def __init__(self, r=None, threshold=0.85, max_scan=500):
        self.r, self.threshold, self.max_scan = r or get_redis(), threshold, max_scan
        self.hits = self.misses = 0

    def _key(self, p): return "llm:cache:" + hashlib.sha256(p.strip().lower().encode()).hexdigest()

    def get(self, prompt):
        k = self._key(prompt)
        raw = self.r.get(k)                                   # 1) exact match
        if raw is None:                                       # 2) semantic match
            q = embed(prompt)
            best, best_s = None, 0.0
            for i, key in enumerate(self.r.scan_iter("llm:cache:*", count=200)):
                if i >= self.max_scan: break
                e = self.r.hget("llm:emb", key)
                if e:
                    s = cosine(q, json.loads(e))
                    if s > best_s: best, best_s = key, s
            if best is not None and best_s >= self.threshold:
                raw = self.r.get(best)
        if raw:
            self.hits += 1; CACHE_HITS.labels(cache_type="semantic").inc()
        else:
            self.misses += 1; CACHE_MISSES.labels(cache_type="semantic").inc()
        HIT_RATIO.labels(cache_type="semantic").set(self.hit_rate)
        return json.loads(raw) if raw else None

    def set(self, prompt, response):
        k = self._key(prompt)
        self.r.set(k, json.dumps(response), ex=ttl_for(prompt))
        self.r.hset("llm:emb", k, json.dumps(embed(prompt)))

    @property
    def hit_rate(self): return self.hits / max(1, self.hits + self.misses)


# ---- complexity classifier (0-100); <30 -> 8B, >=30 -> 70B -------------------
COMPLEX_TERMS = ["analyze", "compare", "comprehensive", "plan", "strategies", "strategy", "recommend",
                 "recommendations", "implications", "rebalancing", "portfolio", "in detail", "decide",
                 "tax", "retirement plan", "review my", "based on current", "complex", "total costs"]
MEDIUM_TERMS = ["explain", "benefits", "how does", "documents", "investment"]


def complexity_score(prompt: str) -> int:
    p = prompt.lower(); n = len(prompt.split())
    score = min(25, n * 1.0)                                         # length (max 25)
    score += 18 * sum(t in p for t in COMPLEX_TERMS if t not in ("tax",)) ** 0.8 if any(t in p for t in COMPLEX_TERMS) else 0
    score += 3 * sum(t in p for t in MEDIUM_TERMS)
    score += 8 if re.search(r"\$\d|\d+k|\d+%|\d+ months", p) else 0   # numbers/money -> reasoning
    score += 6 if p.count(" and ") + p.count(",") >= 2 else 0
    return int(min(100, score))


@dataclass
class Route:
    model: str; score: int; reason: str


def route(prompt: str, threshold: int = 30) -> Route:
    s = complexity_score(prompt)
    return Route("llama-3.1-70b", s, "complex") if s >= threshold else Route("llama-3.1-8b", s, "simple")


def cost(model, tin, tout): return tin / 1000 * MODELS[model]["in"] + tout / 1000 * MODELS[model]["out"]


# ---- self-test / measurement --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", default="test_prompts.csv"); a = ap.parse_args()
    rows = list(csv.DictReader(open(a.prompts)))
    ok = 0; confusion = WordCounter()
    for r in rows:
        d = route(r["prompt"]); good = d.model == r["expected_model"]; ok += good
        confusion[(r["expected_model"], d.model)] += 1
        if not good: print(f"MISROUTE score={d.score:3d} exp={r['expected_model']} got={d.model} :: {r['prompt'][:70]}")
    print(f"\nRouting accuracy: {ok}/{len(rows)} = {ok/len(rows):.0%}")
    print("Confusion:", dict(confusion))

    cache = SemanticCache()
    for r in rows:                                        # first pass fills cache
        if cache.get(r["prompt"]) is None: cache.set(r["prompt"], {"text": "ok"})
    paraphrases = ["what's my account balance", "show my last five transactions", "how can i reset my password",
                   "what are your branch hours", "what is the routing number for a wire transfer"]
    cache.hits = cache.misses = 0
    for q in paraphrases: cache.get(q)
    print(f"Semantic cache on paraphrased queries: {cache.hits}/{len(paraphrases)} hits")


if __name__ == "__main__":
    main()
