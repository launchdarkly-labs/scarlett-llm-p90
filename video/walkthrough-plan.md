# Tutorial walkthrough draft

Status: approved and produced. Generated media stays out of this companion repository; the publishable video lives with the tutorial docs.

Target length: 7 to 8 minutes  
Format: 2560×1440, H.264/AAC, narrated screen recording with open captions  
Article placement: after **What you will build**, replacing the short results video  
Social asset: `social-script.txt`, recorded separately in person

## Shots

1. **The result and the question** (0:00–0:25)
   - Show the pinned 170-task result cards.
   - Establish the reading order: resolution, latency among successes, cost.

2. **Repository setup** (0:25–1:05)
   - Terminal: clone, `uv sync --frozen`, tests, and `.env` copy.
   - Never reveal `.env` contents.

3. **What BFCL supplies** (1:05–1:40)
   - Show the pinned task split and one shortened multi-turn row.
   - Show the official checker boundary without displaying a complete answer.

4. **MCP and skills create the project and config** (1:40–2:25)
   - Show the project/config prompt as a code card.
   - Show current **Agent management → Agents → Completions**.
   - URL: `https://app.launchdarkly.com/projects/llm-p90-experiment/ai-configs/bfcl-model-comparison/variations?env=production&selected-env=production`
   - Show both active variations, equal parameters, neutral message, and attached tools.

5. **Create the paired metrics** (2:25–3:15)
   - Show the context-registration command and the metric/experiment prompt.
   - Show both current metric definitions.
   - URLs:
     - `https://app.launchdarkly.com/projects/llm-p90-experiment/metrics/task-resolution/details?env=production&selected-env=production`
     - `https://app.launchdarkly.com/projects/llm-p90-experiment/metrics/task-resolution-duration-p90/details?env=production&selected-env=production`

6. **Run the paired pilot** (3:15–4:00)
   - Terminal card: one-task command, then 30-task command. Do not rerun either.
   - Show the existing Evaluations run summary and mixed pass/fail rows.
   - URL: `https://app.launchdarkly.com/projects/llm-p90-experiment/ai/evaluations/5d237aa4-98ba-44ea-b9b1-3b7e86d8d94b/runs/704ccd0d-b125-4d90-aac2-a9a992b94cbe?env=production&selected-env=production`

7. **Inspect the draft design** (4:00–4:35)
   - Show the 50/50 design, baseline, treatment, randomization unit, and metrics.
   - URL: `https://app.launchdarkly.com/projects/llm-p90-experiment/experiments/bfcl-model-comparison/design?env=production&selected-env=production`
   - State clearly that the recorded page is the same design after the measured run started.

8. **Prove events arrive before spending** (4:35–5:15)
   - Terminal card: five-row smoke command. Do not rerun it.
   - Show the MCP event-check prompt and current last-seen evidence.
   - Emphasize the test environment.

9. **Read back, then run** (5:15–5:55)
   - Show the MCP preflight prompt.
   - Terminal card: `uv run python run_experiment.py --tasks 170 --confirm-paid-run`.
   - Use the checked-in source log for output. Do not invoke providers.

10. **What the runner records** (5:55–6:20)
    - Code shot: the three `track()` calls.
    - Animate resolved, duration, and within-budget events.

11. **Resolution first** (6:20–6:55)
    - Pinned card: 71/85 versus 63/85.
    - Pinned interval and p-value.

12. **Then p90 among successes** (6:55–7:30)
    - Pinned card: 28.9 versus 29.5 seconds.
    - Show sample sizes 71 and 63 beside assigned counts of 85 and 85.

13. **Use the pattern on your workload** (7:30–7:55)
    - Close on the event contract: assignment, definition of done, resolution, successful duration, time budget.
    - Final card: **Success → latency among successes → cost**.

## Recording rules

- Use current Agent Management terminology and navigation.
- Use the checked-in 170-row log and resource snapshot for every result card.
- Do not show the live results page because it contains one unrelated later exposure.
- Do not run paid commands for B-roll.
- Do not show credentials, operator tokens, SDK keys, or `.env`.
- Keep model-provider names out of narration; required model identifiers may appear in commands and UI.
- Generate narration only after script and shot-plan approval.
