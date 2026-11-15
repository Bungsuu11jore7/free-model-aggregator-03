# Free-model aggregator, variant 3

A single-file Python (stdlib only) aggregator for the make.fun
free-model-aggregator-01 competition. It speaks all three client/provider
wires (OpenAI, Anthropic, Gemini), serves only free models, and translates
requests between wires while keeping the echoed wire fields exact.

## Strategy

1. **Per-requirement shortlisting.** Every request is matched against the
   free models that meet all of its requirements (tools, vision, JSON mode,
   minimum context). An OpenAI-wire JSON request never goes to an
   Anthropic-wire provider, because that wire cannot carry JSON mode.
2. **A pre-computed capacity plan.** Before the first request, the whole
   trial's assignment is fixed: the most constrained requests (fewest
   eligible models) are placed first, weight-2 requests before weight-1,
   and each provider gets a planned budget equal to the smallest per-trial
   budget the practice instances ever showed for it. The top-rated models
   are spent where their rating advantage is largest instead of being
   burned on whichever requests arrive first.
3. **Failure-history-aware failover.** A provider that refuses a call is
   not retried for that request; it is demoted by a fixed rating penalty
   and returns to the top of the order only when it is still the best
   after the penalty. A provider with a scheduled failure call costs one
   tier; a truly spent provider sinks out of use after a few more
   refusals. The plan is advisory: capacity that frees up mid-trial is
   picked up by the same ordering rule.

## Running

`python3 main.py` reads `MAKEFUN_PROMPT_FILE` and `MAKEFUN_ENV_URL`,
registers, serves the workload, and exits when the environment says the
work is done. `python3 main.py --list` prints the catalog document it
registers, byte for byte.

## Trial results

Run with `~/workspace/makefun/run_trial.sh <trial> <bot-dir>`.
