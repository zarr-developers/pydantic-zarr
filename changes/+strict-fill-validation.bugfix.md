The strict `Core`/`Extra` v3 array specs now correctly reject invalid `fill_value`s: non-hex
float strings (e.g. `"garbage"`), out-of-range or non-whole integer fills, booleans for integer
dtypes, malformed complex components, and out-of-range raw bytes. Previously these were
accepted. This is a behavioural change for code that relied on the lax validation.
