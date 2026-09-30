# Workflow throughput audit and user-authorized operational correction

Source: user asked why progress was slow and conditionally authorized pausing Astra
if its endpoints are unavailable. This note changes operational priority, not the
scientific question, frozen acceptance thresholds, budgets, models, efforts or splits.

## Verified observations

Registry window: 2026-09-30 11:48–15:48 UTC. There were 22 registered runs, 9 failed
and 13 process-complete. Registered experiment wall time summed to 288.325 seconds,
about 2.0% of the four-hour window. This sum includes loader/import checks; it is not
training time or a CPU benchmark. API call durations from the three concurrent services
overlap and must not be summed to describe project elapsed time.

Most structured runs consumed zero scientific resources. Historical failures included
Path/string-formatting operator precedence, unterminated source strings, incorrect
interpreter selection, missing imports and context-loader schema mismatch. Some failures
lacked a measured outcome receipt and therefore correctly required review. This prevents
unsafe automatic retries but also exposed weak pre-execution checks.

Astra's primary provider returned HTTP429 DAILY_LIMIT_EXCEEDED, and a backup long
request timed out. The backup subsequently completed review at 15:45:46 UTC with retry
failure count reset to zero. As of that recovery it does not meet the user's conditional
pause condition. Provider limits are not a locally imposed daily token quota.

## Operational corrections and priorities

1. Syntax/compilation and the observed Path-formatting precedence error are checked
   before a Python edit is saved. The selected interpreter also compiles the main
   script before a registered launch. These checks never import or execute scientific
   code, spend simulation budget, or create another scientific review gate.
2. Keep Opus as scientific lead and GPT-5.5 as executor. Preserve substantive failures
   and independent criticism; do not bypass scientific gates or unknown-use accounting.
3. Finish the currently approved source242 identity check and the existing bounded
   converged objective measurement (<=6 solver calls) as soon as its actual prerequisites
   pass. Prefer a useful measurement over another same-purpose report. Batch routine
   bookkeeping; already verified facts need only evidence references unless changed.
4. Any further scientific prerequisite must identify a distinct unresolved validity
   threat and its discriminating check. Do not add metadata-only prerequisites as if
   they were scientific progress. This is not permission to remove genuine gates.
5. Record pending ORIGINAL SAC and pendulum coverage honestly. Once the objective
   contract is settled, move to meaningful control/training comparisons under the
   approved protocol rather than indefinitely continuing this local audit thread.

No evidence in this window justifies upgrading the EC2 instance. Pipeline quality and
handoff overhead are the demonstrated bottlenecks. The current server remains unchanged.
