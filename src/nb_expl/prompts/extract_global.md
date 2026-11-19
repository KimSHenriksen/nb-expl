# Fact sheet for the whole notebook (static analysis, verified)
{{facts}}

# Full notebook
{{code}}

# Task
You have the whole notebook. Extract the claims that only make sense across cells, which a cell-by-cell reading would miss:
- the overall purpose and the end-to-end flow from inputs to outputs (topic "purpose" / "data_flow")
- the order and priority of the matching/allocation steps, and what happens to money that is not matched (topic "business_rule")
- values that change meaning between cells, global state shared between functions, order dependence and non-determinism (topic "risk")
- checks that look like safeguards but cannot actually fail, and checks that are missing (topic "risk" / "invariant")
- invariants that must keep holding after any change (topic "invariant")
- behaviour that differs between dry run and real run (topic "side_effect")

Aim for roughly {{target}} claims. Every claim still needs exact citations.
