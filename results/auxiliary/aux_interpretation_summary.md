# Auxiliary Interpretation Summary

The auxiliary experiments should be framed as diagnostic stress tests.

In `A1`, the model is forced to choose from a false-color answer family. The resulting compliance/conflict-aligned rates therefore measure sensitivity to a constrained answer space, not ordinary open-answer conflict following. In `A2`, the prompt explicitly asks the model to assume the false color family; high compliance is a response to a counterfactual premise and is not directly comparable to C0-C4.

The strongest paper structure is to present A1/A2 after the main results have already established the narrow C0-C4 finding. A1/A2 can then be used to show that prompt format matters and that stronger constraints can increase false-color compliance. They should not be used to inflate the central claim or to argue that the main LLaVA C3 effect is broadly stable.

Suggested wording: "A1/A2 are auxiliary diagnostics rather than primary evidence. They indicate that restricted answer spaces and counterfactual assumptions can elicit high false-color compliance, but the main claim remains based on C0-C4 same-image paired comparisons and the C3 wording boundary control."
