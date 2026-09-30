# Failed quality evaluation — do not use

The original quality selection yielded zero pairs because tokenizer padding was not disabled. NaN-containing raw outputs are retained as failure evidence, not valid JSON metrics. Corrected quality is in ../minilm-short-request-check-v2/quality. Original latency records remain valid and are reused with hashes in v2.
