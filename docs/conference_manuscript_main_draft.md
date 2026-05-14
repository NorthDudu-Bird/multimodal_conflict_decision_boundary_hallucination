# Conference Manuscript Main Draft

This is a first-round conference-style manuscript draft produced from the locked project
evidence. It is intended for structure and argument development before figure,
data-availability, and final venue-formatting passes. Citation keys in this draft are a
first-pass conference bibliography scaffold, with the detailed claim-to-reference map in
`docs/conference_citation_mapping.md` and BibTeX entries in
`docs/references/conference_first_pass.bib`.

## One-Sentence Core Claim

In a controlled car-body primary-colour task, erroneous textual colour cues produce a
limited, same-image conflict-following shift in LLaVA-1.5-7B under specific single-turn
templates, while the effect is absent or near-zero for the other tested models and is
bounded by wording, answer format, prompt factor, colour pair, and residual image-level
confounds.

## Title Candidates

1. **When False Text Changes a Visual Answer: Paired Evidence from Car-Colour VLM Conflicts**
2. **Same-Image Conflict Following in Vision-Language Models under False Colour Cues**
3. **A Controlled Paired Study of Text-Induced Colour Judgement Shifts in VLMs**
4. **How Far Can a False Colour Cue Move a Vision-Language Model?**
5. **Local, Template-Sensitive Conflict Following in Car-Colour Vision-Language Evaluation**

## Abstract

Vision-language models are often evaluated under prompts that assume agreement between
the image and the text. In deployed or adversarial settings, however, textual context can
be wrong, and an erroneous answer under such conflict is difficult to interpret: it may
reflect visual misperception, prompt following, answer-format effects, language priors,
or fragile output parsing. We study this attribution problem in a deliberately narrow
setting: primary car-body colour recognition with false textual colour cues. We build a
300-image balanced evaluation set over six colours and evaluate LLaVA-1.5-7B,
Qwen2-VL-7B-Instruct, and InternVL2-8B under a neutral C0 prompt and a single-turn
C0-C4 conflict prompt family. All three models are faithful under C0 on all 300 images.
In the primary conflict family, LLaVA-1.5-7B shows the only clear conflict-following
shift, with 27/300 outputs in C3 and 10/300 in C4 aligned with the false prompt colour;
Qwen2-VL-7B-Instruct has 1/300 in each of C3 and C4, and InternVL2-8B has 0/300 across
C0-C4. Because all conditions use the same images, the LLaVA cases can be interpreted
as paired answer flips from faithful C0 outputs to false-text-aligned outputs. Controlled
diagnostics show that this shift is not a stable cross-template effect: it weakens under
semantically related C3 rewrites, shrinks under alternative answer formats, depends on
the form of the false text, and is concentrated especially in the white->black route.
Parser, source, visual-clarity, and reproducibility checks reduce several alternative
explanations, while leaving residual local confounds. These results support a local and
conditional account of text-related shifts in this task, not a general claim that VLMs
prioritise text over vision.

## 1. Introduction

Vision-language models (VLMs) are increasingly used in settings where images are
interpreted together with natural-language instructions, descriptions, or context
[Radford2021CLIP; Alayrac2022Flamingo; Li2023BLIP2; Liu2023VisualInstruction;
Liu2023Improved]. A central question for their reliability is how they behave when the
visual evidence and the text disagree. If a prompt contains an erroneous claim about an
image, a model may ignore the claim, correct it, partially incorporate it, or follow it.
A conflict error is therefore not self-explanatory. It may be caused by visual
perception failure, linguistic compliance, answer-space constraints, prompt wording,
output parsing, or some combination of these factors.

Prior work has studied related issues under visual question answering, hallucination
evaluation, language priors, visual-knowledge conflict, object-attribute binding, and
controlled multimodal diagnostics [Goyal2017VQAv2; Agrawal2018DontAssume;
Li2023POPE; Guan2023HallusionBench; Lee2024VLindBench; Liang2025ColorBench;
Yuksekgonul2023ARO; Thrush2022Winoground]. These studies show that cross-modal
conflict is a useful stress test, but they also make the attribution problem sharper.
When the task is broad, the answers are open-ended, or the image pool changes across
conditions, it is hard to know whether a false-text-aligned answer is truly an effect of
the textual cue or simply an ordinary visual or evaluation failure. A useful diagnostic
therefore needs a short evidence chain: stable visual baseline, identical images across
conditions, explicit conflict labels, simple parsing, and controlled boundary checks.

This paper examines that evidence chain in one intentionally restricted task: identifying
the primary body colour of the principal car in a real image. The setting is narrow by
design. It does not test general visual reasoning, vehicle recognition, object relations,
multi-attribute binding, or broad deployment robustness. Instead, it asks whether false
textual colour cues can move a model away from image evidence when the visual attribute
is simple, the answer space is fixed, and the same image is evaluated under neutral and
conflict prompts.

We evaluate three fixed VLMs on a 300-image, six-colour balanced car-image set. The
primary experiment compares a neutral C0 condition with four single-turn conflict
conditions, C1-C4, where the prompt introduces an erroneous colour cue. This design
allows each conflict answer to be paired with the same model's C0 answer on the same
image. We define a response as faithful when it matches the ground-truth car-body
primary colour, and conflict-following when it matches the false prompt colour. The key
unit of evidence is a same-image answer flip from faithful under C0 to conflict-following
under a conflict condition.

The main finding is local and asymmetric. All three models are faithful under neutral
C0 prompting on all 300 images. Under the primary C0-C4 conflict family, LLaVA-1.5-7B
shows a limited but statistically significant conflict-following shift in C3 and C4,
while Qwen2-VL-7B-Instruct and InternVL2-8B remain essentially stable in the same
template family. Further diagnostics show that the LLaVA shift is sensitive to wording,
answer format, prompt factor, and colour pair. The strongest supported conclusion is
therefore not that text generally overrides vision, but that one tested model exhibits
a bounded same-image shift under particular false-colour prompt forms.

The contributions are:

- We introduce a controlled paired evaluation setup for a narrow image-text conflict
  question, using a balanced car-body colour task and identical images across neutral
  and conflict conditions.
- We report a model- and template-specific conflict-following shift for LLaVA-1.5-7B,
  grounded as same-image faithful-to-conflict answer flips rather than unpaired aggregate
  differences.
- We provide boundary diagnostics showing that the observed shift is wording-sensitive,
  answer-format-sensitive, prompt-factor-sensitive, and concentrated in specific colour
  routes.
- We separate primary evidence, auxiliary stress tests, controlled diagnostics, validity
  checks, and extension diagnostics so that stronger prompt-compliance phenomena do not
  overwrite the narrower C0-C4 claim.

[FIGURE PLACEHOLDER: Figure 1 should summarize the evidence chain: image -> C0 neutral
answer -> C1-C4 false-colour prompts -> paired flip definition -> boundary diagnostics.]

## 2. Related Work

### Multimodal conflict and hallucination evaluation

Multimodal hallucination and conflict benchmarks provide the closest evaluation context
for this work. Existing studies ask whether VLM outputs remain grounded when visual
evidence, textual context, or world knowledge are unreliable [Li2023POPE;
Guan2023HallusionBench; Lee2024VLindBench]. These benchmarks motivate controlled
conflict testing, but many operate over broad question types, object categories, or
free-form outputs. Our study differs by narrowing the task to one visual attribute and
by treating same-image paired flips as the primary unit of interpretation.

### Language priors and diagnostic separation

Language priors can produce correct-looking or plausible-looking answers without
adequate visual grounding [Goyal2017VQAv2; Agrawal2018DontAssume]. However, a
false-text-aligned answer is not automatically a pure language-prior effect. It can also
emerge from poor image visibility, ambiguous labels, response-format pressure, or parser
behaviour. For this reason, recent diagnostic work has emphasized separating
perception, knowledge, bias, and response-format effects [Lee2024VLindBench]. We adopt
this separation as an evaluation principle, but we do not attempt to assign broad
model-level blindness or bias scores.

### Colour, attributes, and object grounding

Colour is an interpretable attribute, but it is not a trivial probe. Prior benchmarks
on colour understanding, compositionality, and attribute binding show that VLM behaviour
can vary with object category, colour distribution, visual ambiguity, and prompt form
[Liang2025ColorBench; Yuksekgonul2023ARO; Thrush2022Winoground]. This motivates using
colour as a controlled observation window while avoiding a general colour-perception
claim. In our setting, car-body primary colour is selected because it is visually
inspectable and has a compact label space; the study does not propose a general colour
benchmark.

### Position of this work

This paper is best read as a small but tightly controlled behavioural analysis. It does
not introduce a new model, training method, mitigation, or broad leaderboard. Its
distinction is the combination of a fully faithful neutral baseline, identical images
across conditions, paired flip accounting, and explicit boundary diagnostics around the
observed conflict-following cases.

## 3. Task and Experimental Setup

### 3.1 Task definition

The task is to identify the primary body colour of the principal car in an image. The
canonical labels are black, blue, green, red, white, and yellow, with an additional
`other` output option for prompts where a response falls outside the target labels. A
response is faithful if the parsed label equals the ground-truth car-body primary colour.
A response is conflict-following if the parsed label equals the false colour introduced
by the conflict prompt. A same-image paired flip occurs when the same model answers the
same image faithfully under C0 but follows the false prompt colour under a conflict
condition.

### 3.2 Evaluation set

The main evaluation set contains 300 real car images, balanced across six true colours
with 50 images per class. The set combines 93 StanfordCars images and 207 VCoR images,
but source identity is not treated as a primary experimental factor. Both sources are
used as mixed-source real vehicle-image inputs after cropping, and source identity is
retained only for sanity checks and limitation analysis.

[TABLE PLACEHOLDER: Table 1 should report the 300-image distribution by true colour and
source composition. Use `data/metadata/balanced_eval_set/balanced_eval_set_summary.json`
as source data.]

### 3.3 Models

We evaluate three fixed VLMs: LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, and InternVL2-8B.
The comparison is local to these checkpoints, prompts, and images. It should not be
interpreted as a model-family ranking, a scale law, or a claim about global robustness.
We cite the corresponding model reports only to identify the tested model families, not
as evidence for the robustness conclusions in this paper [Liu2023VisualInstruction;
Liu2023Improved; Wang2024Qwen2VL; Chen2023InternVL; Chen2024InternVL2].

### 3.4 Prompt conditions

The primary experiment uses one neutral condition, C0, and four single-turn conflict
conditions, C1-C4. C0 asks for the car-body primary colour without introducing a false
colour cue. C1-C4 introduce erroneous textual colour information with different wording
strengths and framing. Each model is evaluated on the same 300 images under each
condition. This creates a matched model-condition-image grid and allows paired analysis
against the model's own C0 answer.

[TABLE PLACEHOLDER: Table 2 should summarize C0-C4 prompt roles without reproducing
every full prompt. Full prompts can be placed in appendix or supplementary material.]

### 3.5 Parsing and metrics

Model outputs are parsed into the six target colour labels or `other`. The primary
C0-C4 outputs remain in the base single-label regime: the parser audit reports 4500
main parsed rows with no parse errors, refusals, corrections, or other-wrong outputs.
The main metrics are faithfulness rate, conflict-following rate, answer-flip rate,
faithful retention rate, and paired exact tests against C0. Holm correction is used for
predefined multiple comparisons.

## 4. Results

### 4.1 Neutral prompting establishes a stable visual baseline

The primary evidence chain begins with the neutral C0 baseline. All three models answer
all 300 images faithfully under C0, yielding faithful=300/300 and conflict_aligned=0/300
for LLaVA-1.5-7B, Qwen2-VL-7B-Instruct, and InternVL2-8B. The main C0-C4 table also
contains no refusals, parse errors, or other-wrong outputs. This establishes that the
task is not dominated by baseline colour-recognition failure under neutral prompting.

[TABLE PLACEHOLDER: Table 3 should report main C0-C4 metrics from
`results/main/table1_main_metrics.csv`. Highlight LLaVA C3/C4 but keep all model rows.]

### 4.2 Primary conflict prompts produce a limited LLaVA-specific shift

Under the primary C0-C4 conflict family, most model outputs remain visually faithful.
Qwen2-VL-7B-Instruct produces no conflict-following outputs in C1 or C2 and only one
in each of C3 and C4. InternVL2-8B produces no conflict-following outputs across C0-C4.
LLaVA-1.5-7B is the only model with a clear primary shift: 1/300 in C1, 3/300 in C2,
27/300 in C3, and 10/300 in C4. The LLaVA C3 rate is 9.00% with a 95% confidence
interval of 6.26%-12.78%; the C4 rate is 3.33% with a 95% confidence interval of
1.82%-6.03%. The C3 and C4 increases are significant in paired tests against C0 and
in matched-image comparisons against the stable models.

[FIGURE PLACEHOLDER: Figure 2 should show conflict-following rates across C0-C4 for
the three models, with confidence intervals and a caption defining the interval method.]

### 4.3 Same-image paired flips make the attribution narrower

Because all conditions use the same images and all C0 outputs are faithful, the LLaVA
C3 and C4 conflict-following outputs can be written as same-image paired flips. LLaVA
has 27/300 faithful-to-conflict flips in C3 and 10/300 in C4, corresponding to faithful
retention rates of 91.00% and 96.67%, respectively. Qwen2-VL-7B-Instruct has one such
flip in each of C3 and C4. InternVL2-8B has none across C1-C4.

This paired interpretation is stronger than reporting condition-level rates alone. It
does not prove a general causal mechanism, but it removes an important source of
cross-sample ambiguity: each changed answer is tied to the same visual evidence under
a changed prompt.

[FIGURE PLACEHOLDER: Figure 3 should visualize paired C0-to-conflict transitions or
answer-flip rates, using `results/main/paired_flip_metrics.csv`.]

### 4.4 Auxiliary diagnostics show compliance under stronger constraints

A1/A2 are retained as auxiliary diagnostics. A1 restricts the answer space toward the
false colour family, and A2 asks the model to operate under a counterfactual false-colour
assumption. These conditions produce much higher compliance rates: Qwen2-VL-7B-Instruct
has 55.67% in A1 and 90.67% in A2; LLaVA-1.5-7B has 85.33% in A1 and 100.00% in A2;
InternVL2-8B has 73.67% in A1 and 100.00% in A2.

These results show that stronger answer-space constraints and counterfactual assumptions
can push all tested models toward the false colour family. They do not replace the
open-answer C0-C4 evidence chain, because the intervention changes the answer space and
task assumption.

### 4.5 Controlled diagnostics bound the main claim

The controlled diagnostics show that the LLaVA C3/C4 observation is conditional rather
than stable across prompt variants. First, C3 wording robustness substantially weakens
the effect. LLaVA drops from 27/300 under canonical C3 to 5/300 under C3-v2 and 0/300
under C3-v3. The wording variants no longer show Holm-significant LLaVA-vs-stable-model
differences.

Second, the effect is not uniform across colours. Among LLaVA's 27 C3 flips, 20 are
white->black, 3 are black->white, and 4 are blue->red. C4 is also concentrated in the
white->black route, with 8/10 flips in that route and 2/10 in black->white. Thus, the
9.00% C3 result should not be described as a broad colour-task susceptibility.

Third, answer format changes the observed rate. For LLaVA-1.5-7B, canonical C3 remains
larger than matched free-answer C3 at 7/300, multiple-choice C3 at 4/300, and yes/no
false-claim probing at 4/300. Fourth, prompt factorization shows that false text is not
a single monolithic intervention. Quoted claims and indirect hints remain weak, whereas
title/prefix framing and no-correction presupposition can be much stronger, including
for Qwen and InternVL2. These diagnostics constrain interpretation; they should not be
rewritten as a new prompt-engineering mainline.

[FIGURE PLACEHOLDER: Figure 4 should combine wording robustness, colour-pair split, and
answer-format/factor diagnostics as boundary evidence. Keep this figure secondary to
the primary paired C0-C4 figure.]

### 4.6 Validity checks reduce, but do not eliminate, alternative explanations

Parser audit reduces the risk that conflict-following counts are inflated by fragile
label mapping. In the primary C0-C4 experiment, all 4500 rows parse successfully, and
the outputs stay within the base single-label regime. Source-stratified checks reduce
the concern that the main pattern is confined to one source subset: under C0, all three
models are faithful in both StanfordCars and VCoR; under C3, LLaVA conflict-following
appears in both source groups, with 13/93 in StanfordCars and 14/207 in VCoR. This check
supports source robustness as a sanity check, not a source-generalization benchmark.

The visual clarity audit reviews 42 target flip rows and 42 matched faithful controls.
Most target rows are visually inspectable, with 38/42 clear compared with 39/42 controls.
This weakens a global difficult-image explanation. However, visual confound flags are
more frequent among target rows than controls, at 11/42 versus 4/42. Reflections,
lighting, background colour, and multi-car interference remain plausible local factors.
Finally, the reproducibility audit reports that all locked experimental artifacts match
the snapshot; the only non-blocking difference is a writing-facing integrated summary.

### 4.7 Extension diagnostics identify a separate dialogue-context boundary

The multi-turn diagnostic tests a different interaction regime and is not part of the
primary single-turn evidence chain. LLaVA and Qwen remain near zero in the tested
multi-turn conditions, while InternVL2 rises to 21.33% in MT2 and 74.67% in MT3. This
shows that repeated dialogue context can create a strong model-specific vulnerability,
but it should not be used to reinterpret the single-turn C0-C4 result.

## 5. Discussion

The central result is a local paired observation. In a controlled car-body colour task,
all three models are fully faithful under neutral prompting, and most conflict-condition
outputs remain faithful. Within the primary single-turn prompt family, however,
LLaVA-1.5-7B shows limited same-image conflict-following under canonical C3 and C4. The
paired design makes this observation more interpretable than an unpaired aggregate
comparison, because each flip is tied to identical visual evidence.

The result should not be generalized beyond its evidence. The C3 shift weakens under
semantically related rewrites, shrinks under alternative answer formats, and concentrates
in a small set of colour routes. These findings suggest that prompt surface form and
colour-pair structure are part of the measured phenomenon. The study therefore supports
a bounded behavioural account rather than a stable cross-template law.

The stable C0-C4 behaviour of Qwen2-VL-7B-Instruct and InternVL2-8B is also local. It
supports the statement that these two models are essentially stable in the present
primary template family. It does not imply that they are globally robust to all forms of
misleading text. Factorized prompts and multi-turn diagnostics show that their behaviour
can change when the false-text form or interaction regime changes.

The main value of this study is methodological as much as numerical. It separates the
primary paired evidence chain from auxiliary stress tests, controlled diagnostics,
validity checks, and extension diagnostics. This separation prevents stronger but
different compliance phenomena from being folded into the main single-turn claim. It
also gives reviewers a clearer path for judging what the evidence supports and where it
stops.

## 6. Limitations

The task scope is intentionally narrow. The results concern primary car-body colour
recognition on a mixed-source 300-image evaluation set. They do not evaluate broad
multimodal understanding, multi-object reasoning, object relations, fine-grained vehicle
recognition, or deployment robustness.

The model scope is limited to three checkpoints. Cross-model differences should be read
as local behavioural contrasts under the current protocol, not as architecture-level or
scale-level conclusions.

The prompt scope is also limited. The clearest result occurs under canonical C3 and,
secondarily, C4. Wording, answer format, and prompt factorization change the observed
rates. Future work should test larger prompt families before treating any false-text
effect as stable.

The visual clarity audit reduces but does not eliminate visual-confusion explanations.
Target flip rows are mostly inspectable, but local confounds are more common among flips
than controls. The current evidence supports a text-related shift in specific cases; it
does not prove that every flip is free of visual ambiguity.

Finally, the current bibliography is a first-pass conference scaffold rather than a
venue-formatted final reference list. Before submission, the references should be
converted to the target style, checked for official proceedings versions where
available, and trimmed so that each citation continues to support a specific local
claim.

## Working Citation Resources

The first citation pass produced two companion files:

- `docs/conference_citation_mapping.md` records which references support each manuscript
  segment and marks whether the support is direct, contextual, or boundary-setting.
- `docs/references/conference_first_pass.bib` contains the current BibTeX scaffold for
  conference-template conversion.

## 7. Conclusion

This paper studies whether erroneous textual colour cues can shift VLM judgments away
from image evidence in a controlled car-body primary-colour task. The answer is
conditional. All three tested models are faithful under neutral C0 prompting. In the
primary single-turn C0-C4 prompt family, LLaVA-1.5-7B shows limited but significant
same-image conflict-following under canonical C3 and C4, while Qwen2-VL-7B-Instruct and
InternVL2-8B remain essentially stable in that primary family. Controlled diagnostics
show that the shift is wording-, format-, factor-, and colour-pair-sensitive, and
validity checks reduce but do not remove alternative explanations. The strongest
supported conclusion is therefore a local and bounded one: false textual cues can shift
one tested model in specific prompt settings, but the evidence does not support a
general text-over-vision claim.

## Claim-Evidence Map

| Claim | Evidence source | Status |
| --- | --- | --- |
| All three models are faithful under C0. | `results/main/table1_main_metrics.csv`; `results/final_result_summary.md` | Supported |
| LLaVA shows limited conflict-following under canonical C3/C4. | `results/main/table1_main_metrics.csv`; `results/main/main_key_tests.csv` | Supported |
| LLaVA C3/C4 cases can be interpreted as same-image paired flips. | `results/main/paired_flip_metrics.csv`; `results/main/paired_flip_summary.md` | Supported |
| Qwen and InternVL2 are stable in the primary C0-C4 template family. | `results/main/table1_main_metrics.csv` | Supported within current protocol |
| The LLaVA C3 effect is wording-sensitive. | `results/robustness/prompt_boundary_summary.md` | Supported |
| The LLaVA effect is concentrated in white->black and related achromatic routes. | `results/color_split/color_split_summary.md` | Supported |
| Parser artifacts are low risk in the main experiment. | `results/parser/label_mapping_audit.md` | Supported |
| Visual confusion is fully ruled out. | `results/audit/visual_clarity_audit_summary.md` | Not supported; must remain a limitation |
| The result generalizes to VLMs broadly. | Current evidence | Not supported |

## Next-Pass TODOs

- Convert the first-pass BibTeX scaffold into the final venue style and verify
  proceedings metadata for accepted versions.
- Convert figure placeholders into a concrete figure plan before invoking
  `nature-figure`.
- Add Data Availability, Code Availability, and third-party data/source statements with
  `nature-data`.
- After citations and figures are added, run `nature-polishing` for paragraph flow,
  title compression, abstract tightening, and conference-style language cleanup.
