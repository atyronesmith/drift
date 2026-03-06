# Collaborative Intelligence in AI: Evaluating the Performance of a Council of AIs on the USMLE

**Paper**: PLOS Digital Health 4(10): e0000787, October 9, 2025
**Authors**: Shaikh, Y., Jeelani-Shaikh, Z.A., Jeelani, M.M., Javaid, A., Mahmud, T., Gaglani, S., et al.
**DOI**: https://doi.org/10.1371/journal.pdig.0000787
**Code**: https://github.com/councilofai/project-saru

---

## What This Paper Does

Creates a "Council of AIs" — five GPT-4 instances that iteratively deliberate on medical exam questions until reaching consensus. Tests on 325 USMLE questions across all three exam steps. Key argument: response variability, usually seen as a flaw, becomes a strength when harnessed through structured deliberation.

## Architecture

1. Question sent to 5 GPT-4 instances (AI_A through AI_F), each with temperature=1, top_p=1
2. A **Facilitator algorithm** (also GPT-4) checks for consensus
3. If no consensus: Facilitator summarizes each agent's reasoning, formulates a clarifying question about the disagreement, and sends back to all agents with the original question
4. Steps 2-3 repeat until unanimous consensus
5. All state reset between questions (no cross-contamination)

Each agent prompted with: domain knowledge scope, structured reasoning pattern (RESTATE -> PLAN -> CLINICAL REASONING -> CONCLUDE -> RESPOND), and awareness of council membership.

## Key Results

**Overall Accuracy**: 94.7% (308/325 correct)
- Step 1 (basic sciences): 96.8%
- Step 2 CK (clinical knowledge): 92.7%
- Step 3 (independent practice): 94.3%

**vs Single Instance**: Individual GPT-4 instances correct 78.2% of the time (all 5 agree). Council deliberation lifts this to 94.7%.

**vs Majority Voting**: Council (94.7%) outperforms simple majority vote (91.7%) by 3 percentage points.

**Error Correction Power**: When initial majority vote was wrong (19 questions), deliberation corrected 10 of them (53%). Odds ratio: 5.0x more likely to change from incorrect to correct than correct to incorrect after deliberation.

**Critical constraint**: The Council *never* achieved a correct consensus when all 5 members initially answered incorrectly. At least one correct initial response was required to guide successful deliberation.

## Semantic Entropy Findings

Calculated Shannon entropy over response distributions at each deliberation round:
- Entropy consistently decreased with each round, approaching zero (full consensus)
- This held even for ultimately incorrect consensus answers
- **Two-phase pattern**: Fast entropy drop in rounds 1-2, then slow convergence. Questions requiring 5+ rounds show erratic entropy trajectories.
- **4-5 round transition point**: Below this, entropy decline is steep and predictable (slopes -0.29 to -0.09). Above this, slopes flatten to -0.07 to -0.02, indicating genuinely difficult reasoning challenges.

## Deliberation Statistics

- 78% of questions achieved immediate unanimous consensus (no deliberation needed)
- 22% required deliberation (at least one member disagreed)
- Average rounds to consensus: 3.6 (Step 1), 4.1 (Step 2 CK), 2.4 (Step 3)
- Most questions needing deliberation (71-85%) resolved in 2 rounds
- No correlation (r^2=0) between number of initial disagreements and rounds needed

## Deliberation Can't Fix Universal Ignorance

Of 72 questions requiring deliberation:
- 54 (75%) reached correct consensus
- 18 (25%) reached incorrect consensus
- When all members were initially wrong, deliberation always failed — the Council needs at least one correct perspective to converge correctly.

## Relevance to Drift

**Directly relevant — this is the "variability as feature" argument:**

- Their core thesis mirrors our research question from the opposite direction. We measure how variance *hurts* pipeline reliability; they show how variance *helps* when properly harnessed through deliberation.
- The **semantic entropy measurement** is a more principled version of our edit distance metrics. Entropy captures semantic diversity; edit distance captures textual diversity. For our structured JSON outputs, these should correlate closely.
- The **entropy convergence pattern** (fast initial drop, then slow) maps to what we might expect from iterative pipeline correction: early structured constraints absorb most variance, but remaining semantic drift is harder to eliminate.
- Their **temperature=1** finding is notable: they deliberately maximize variance to get diverse perspectives. Our temperature=0 experiments show that infrastructure alone introduces enough variance for non-trivial diversity (10/10 unique at Vertex).
- The **"at least one correct response needed"** constraint is crucial for our pipeline design: if step 1's output is fundamentally wrong (not just textually variant), no amount of downstream processing can fix it. This supports spending on a high-quality step 1 model.
- Their **Facilitator pattern** (summarize -> question -> re-query) is a fan-out-and-synthesize topology — exactly what DeepMind's paper identifies as the most effective coordination pattern.
- The **5.0x odds ratio** favoring self-correction suggests that multi-agent deliberation could be a powerful defense against both variance *and* errors in our pipeline experiments.

**Design implications:**
- A "Council" step in our pipeline — fan out to N instances, deliberate, synthesize — could absorb variance more effectively than our current linear chain
- But the cost is significant: 3-7 minutes per question with 5 agents and 3+ deliberation rounds
- Their finding that 78% of questions need no deliberation suggests a hybrid: fast-path for agreement, slow-path deliberation only when variance exceeds a threshold
