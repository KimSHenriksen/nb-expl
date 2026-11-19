# Notebook outline (all cells)
{{outline}}

# Fact sheet: the cells below, plus definitions from the whole notebook (static analysis, verified)
{{facts}}

# Code to analyse: cells {{cells}}
{{code}}

# Task
Extract claims about the code above (cells {{cells}} only). Cover, where present:
- the purpose of each cell in business terms (topic "purpose")
- inputs, parameters and outputs (topics "input", "parameter", "output")
- business rules: matching, allocation order, priorities, tolerances, write-offs, exclusions, currency conversion (topic "business_rule" or "calculation")
- filters and joins that drop or duplicate rows, often silently (topic "filter" or "data_quality")
- how data moves between variables and cells (topic "data_flow")
- dependencies on things outside this file (topic "dependency")
- side effects: writes, notifications, logging (topic "side_effect")
- dead or commented-out code and what it suggests (topic "dead_code")
- risks and things that could go wrong (topic "risk")
- invariants the code relies on or checks, e.g. amounts that must balance (topic "invariant")

Aim for roughly {{target}} claims. Prefer fewer, accurate claims over many vague ones.
