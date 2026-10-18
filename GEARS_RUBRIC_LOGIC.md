# RoboGyn-AI v0.1 — GEARS Rubric & Assessment Logic Specification

## 1. Methodological Foundation
**GEARS** (*Global Evaluative Assessment of Robotic Skills*, Goh et al., 2012) is a validated clinical rating instrument that evaluates technical competence in robotic surgery across six domains.

In **RoboGyn-AI v0.1**, computer vision (dense Farneback optical flow) and human GEARS ratings serve complementary, scientifically distinct roles:
- **Human GEARS Ratings (Surgeon Self & Independent Expert):** Measure clinical surgical skill, tissue handling quality, and operative judgment.
- **Computer Vision Motion Analysis:** Measures objective, observable video features (task duration, active vs. idle motion ratio, mean/peak optical flow magnitude, and high-intensity motion bursts).
- **Design Rule:** Version 0.1 never replaces human GEARS scoring with an unvalidated autonomous AI score. Instead, it pairs blinded human ratings with objective motion indicators to guide targeted video review and deliberate practice.

---

## 2. The 6 GEARS Domains & Clinical Behavioral Anchors

Each domain is scored on a **1 to 5 Likert scale** (Total Score range: **6 to 30 points**). Scores **2** and **4** represent intermediate performance between the anchored levels below:

| Domain ID | GEARS Domain | Score 1 (Novice) | Score 3 (Intermediate) | Score 5 (Expert) |
| :--- | :--- | :--- | :--- | :--- |
| `depth_perception` | **Depth Perception** | Constantly overshoots target, wide swings, slow to correct | Some overshooting or missing of target, but quick to correct | Accurately directs instruments in the correct plane to target |
| `bimanual_dexterity` | **Bimanual Dexterity** | Uses only one hand, ignores non-dominant hand, poor coordination between hands | Uses both hands, but does not optimize interaction between hands | Expertly uses both hands in a complementary manner to provide optimal exposure and traction |
| `efficiency` | **Efficiency** | Inefficient efforts; tentative movements; constantly changing focus or persisting without progress | Slow, but planned movements are reasonably organized | Confident, efficient and safe conduct, maintains focus on task, fluid progression |
| `force_sensitivity` | **Force Sensitivity** | Rough moves, tears tissue, injures adjacent structures, poor control, frequent suture breakage | Handles tissue reasonably well, minor trauma to adjacent tissue, rare suture breakage | Applies appropriate tension, negligible injury to adjacent structures, no suture breakage |
| `autonomy` | **Autonomy** | Unable to complete entire task, even with verbal guidance | Able to complete task safely with moderate guidance | Able to complete task independently without prompting |
| `robotic_control` | **Robotic Control** | Consistently does not optimize view, hand position, or colliding instruments | View is sometimes not optimal; occasionally needs to clutch or adjust arm position | Controls camera and hand position optimally and without collision |

---

## 3. Blinded Rating Workflow (Option B Role Switcher)

To prevent **anchoring bias** between the operating surgeon and the independent expert rater:
1. **Surgeon View (`🧑‍⚕️ Surgeon View`):**
   - The surgeon uploads the de-identified operative video, selects the 30–60s suturing segment, runs the optical flow analysis, and locks their **6-domain GEARS Self-Assessment**.
   - Expert ratings are hidden on this view.
2. **Blinded Expert View (`👩‍🏫 Expert View (Blinded)`):**
   - The independent expert reviews the pre-selected suturing segment and objective motion profile.
   - Both the **Surgeon's Self-Assessment scores** and (by default) the **Surgeon's identity / training level** are hidden while the expert grades.
3. **Combined Review & Admin View (`📊 Combined Review / Admin`):**
   - Unlocks automatically once both ratings are locked (or via Admin override) to display the calibration gap analysis and rubric-based feedback.

---

## 4. Automated Rubric & Calibration Logic

### A. Self–Expert Calibration Formula
For each domain $i \in \{1, \dots, 6\}$:
$$\Delta_i = \text{Self Score}_i - \text{Expert Score}_i$$

| Difference ($\Delta_i$) | Calibration Classification | Clinical Interpretation |
| :---: | :--- | :--- |
| $\Delta_i \ge +2$ | **Overestimated (+2 or more)** | Significant blind spot; trainee perceives high competence where expert observed technical deficits. |
| $\Delta_i = +1$ | **Slightly Overestimated (+1)** | Mild over-confidence; review expert coaching note for subtle refinements. |
| $\Delta_i = 0$ | **Well Calibrated (Exact Match)** | Accurate self-awareness aligned with expert standard. |
| $\Delta_i = -1$ | **Slightly Underestimated (-1)** | Mild under-confidence; performance exceeded trainee's self-perception. |
| $\Delta_i \le -2$ | **Underestimated (-2 or more)** | Trainee was overly self-critical relative to objective expert evaluation. |

The overall **Mean Domain Calibration Gap** is computed as:
$$\text{Mean Absolute Gap} = \frac{1}{6} \sum_{i=1}^{6} |\Delta_i|$$

### B. Practice Priority Trigger Rule
A GEARS domain is automatically flagged as a **Suggested Practice Priority** when:
$$\text{Expert Score}_i \le 3 \quad \text{OR} \quad \text{Self Score}_i \le 2$$

When triggered, the engine outputs:
1. **Domain-Specific Deliberate Practice Drill:** Actionable technical exercise tailored to robotic gynecologic suturing (e.g., vaginal cuff mucosal approximation, visual haptics, non-dominant traction, master-clutching).
2. **Targeted Video Review Tip:** Directs the surgeon to specific features on the optical-flow timeline.

### C. Objective Video-Motion Adjunct Rules
Alongside the GEARS rubric, the optical flow analyzer flags two objective motion patterns for timestamp inspection:
1. **Operative Flow / Idle Ratio Rule:**
   - Active motion threshold: $\tau_{\text{active}} = \max(0.25\text{ px/step},\; 0.30 \times \mu_{\text{flow}})$.
   - If $\text{Idle Time} > 30\%$, the system flags **Elevated Pause/Idle Ratio**, prompting review of low-motion valleys for needle re-orientation delays or exposure struggles (*Efficiency* & *Autonomy* adjunct).
2. **Motion Variability & High-Intensity Burst Rule:**
   - Burst threshold: $\tau_{\text{burst}} = \mu_{\text{flow}} + 1.5 \sigma_{\text{flow}}$.
   - If the coefficient of variation $\text{CV} = \sigma_{\text{flow}} / \mu_{\text{flow}} > 0.65$ or $\text{Burst Count} \ge 4$, the system flags **Motion Spikes Detected**, prompting review of peak timestamps for target overshooting, abrupt knot cinching, or mid-bite camera shifts (*Depth Perception*, *Force Sensitivity*, & *Robotic Control* adjunct).
