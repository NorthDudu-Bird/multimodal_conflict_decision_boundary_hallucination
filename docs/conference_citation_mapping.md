# Conference Citation Mapping

This file records the first-pass `nature-academic-search` + `nature-citation` mapping
for `docs/conference_manuscript_main_draft.md`. The goal is not to maximize citation
count; it is to keep each background claim tied to a specific, conservative source.

## Source Hierarchy Used

1. Official proceedings or publisher pages when available.
2. arXiv records for preprints, model reports, or papers whose proceedings version is
   not needed for the current draft scaffold.
3. No blog or secondary-source citations were used for manuscript support.

## Segment-to-Citation Map

| Manuscript segment | Local claim being supported | Citation keys | Support grade |
| --- | --- | --- | --- |
| Introduction, paragraph 1 | Modern VLMs combine image encoders, language models, and natural-language prompts/instructions. | `Radford2021CLIP`; `Alayrac2022Flamingo`; `Li2023BLIP2`; `Liu2023VisualInstruction`; `Liu2023Improved` | Contextual background |
| Introduction, paragraph 2 | Language priors, hallucination, conflict, and compositional binding are established diagnostic concerns. | `Goyal2017VQAv2`; `Agrawal2018DontAssume`; `Li2023POPE`; `Guan2023HallusionBench`; `Lee2024VLindBench`; `Liang2025ColorBench`; `Yuksekgonul2023ARO`; `Thrush2022Winoground` | Direct for problem framing, not direct evidence for this paper's results |
| Related Work: Multimodal conflict and hallucination evaluation | Existing benchmarks test grounding under unreliable visual/textual/world-knowledge settings, often across broad tasks. | `Li2023POPE`; `Guan2023HallusionBench`; `Lee2024VLindBench` | Direct contextual comparison |
| Related Work: Language priors and diagnostic separation | VQA and LVLM work motivates separating language priors from visual perception and other confounds. | `Goyal2017VQAv2`; `Agrawal2018DontAssume`; `Lee2024VLindBench` | Direct for diagnostic principle |
| Related Work: Colour, attributes, and object grounding | Colour and attribute/compositional binding are known nontrivial probes for VLMs. | `Liang2025ColorBench`; `Yuksekgonul2023ARO`; `Thrush2022Winoground` | Direct contextual comparison |
| Models | Model-family identification for LLaVA, Qwen2-VL, and InternVL/InternVL2. | `Liu2023VisualInstruction`; `Liu2023Improved`; `Wang2024Qwen2VL`; `Chen2023InternVL`; `Chen2024InternVL2` | Identification only |

## Boundary Notes

- The cited VQA and hallucination literature supports motivation and framing; it does
  not by itself support this paper's same-image paired-flip claims.
- The model citations identify the model families and should not be used as evidence
  for model-family rankings.
- `ColorBench`, ARO, and Winoground justify treating colour/attribute binding as
  meaningful diagnostics, but this paper remains a car-body primary-colour study rather
  than a general colour benchmark.
- A specific "CAB" citation was not added in this pass because the candidate acronym
  was ambiguous in the literature search. The manuscript now cites verified colour,
  compositionality, and attribute-binding sources instead.

## Sources Checked

- arXiv: https://arxiv.org/abs/2103.00020
- arXiv: https://arxiv.org/abs/2204.14198
- arXiv: https://arxiv.org/abs/2301.12597
- CVF/arXiv: https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html and https://arxiv.org/abs/1612.00837
- arXiv: https://arxiv.org/abs/1712.00377
- arXiv: https://arxiv.org/abs/2204.03162
- arXiv: https://arxiv.org/abs/2210.01936
- arXiv: https://arxiv.org/abs/2304.08485
- arXiv: https://arxiv.org/abs/2310.03744
- arXiv: https://arxiv.org/abs/2305.10355
- arXiv: https://arxiv.org/abs/2310.14566
- arXiv: https://arxiv.org/abs/2312.14238
- arXiv: https://arxiv.org/abs/2404.16821
- arXiv: https://arxiv.org/abs/2406.08702
- arXiv: https://arxiv.org/abs/2409.12191
- arXiv: https://arxiv.org/abs/2504.10514
