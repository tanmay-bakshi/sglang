# Upstream Merge Section 8 Rung-3 Re-bless Receipt

## Verdict

- Ruling: ladder ruling #5, adopt upstream's `weight_versions` response schema
  as-is.
- Qualified executable revision:
  `881d1cbdfe0d2477d3689328a1e3d20827418928`.
- Merge revision: `0bdf639686b01fedc0db2c25ee7e3b62f1001c63`.
- Pinned upstream parent:
  `6afb5e17712e2e90b60ba8456ca893e529316869`.
- Evidence root:
  `/workspace/upstream-merge-20260830/section8-881d1cbdfe/ruling5`.
- Rung 3 result: `PASS`, `28/28` byte-exact cases under the deliberately
  re-blessed reference.

Gate 0 remains sealed at
`66c47a38863cdb7230d76ad4b0d190733bf13858` and was not rerun. Rungs 1 and 2
remain sealed and were not repeated. This receipt changes no executable source
and does not normalize or suppress any response field.

## Deliberate reference update

The original Gate-0 reference remains intact at
`/workspace/upstream-merge-20260830/section8-881d1cbdfe/corpus/reference.json`
with SHA-256
`c9a9fbc4b46a9773e787001b038d64a8f64509c9f0ef60abc60cec2eed9846d4`.

The authoritative merged capture remains intact at
`/workspace/upstream-merge-20260830/section8-881d1cbdfe/corpus/candidate.json`
with SHA-256
`a5ad63d7728f96322bc3916f5c5184b663654109a2c97b33047ac75b041d2b3c`.
It was copied byte-for-byte to the separately named ruling-5 reference:

`/workspace/upstream-merge-20260830/section8-881d1cbdfe/ruling5/reference-weight-versions.json`

The ruling-5 reference SHA-256 is also
`a5ad63d7728f96322bc3916f5c5184b663654109a2c97b33047ac75b041d2b3c`.
The old reference was not overwritten.

## Exact delta proof

The sealed proof performs two independent checks:

1. the ruling-5 reference bytes equal the authoritative merged capture bytes;
2. after selecting the same deterministic fields as the corpus comparator and
   deleting `metadata.weight_versions` from every new response, the result is
   byte-identical to the Gate-0 reference.

All 28 deleted values are exactly:

```json
[{"version":"default","start":0,"end":1}]
```

The proof therefore establishes that rendered chat-template bytes, prompt
token IDs, generated output, reasoning content, usage values, and every other
stable response field are unchanged. The adopted field is present in every
case; it is neither conditionally emitted nor comparator-normalized.

- Proof receipt SHA-256:
  `569e2e24d49df00dbfbd3e4b36701edbe5ef88d3217ef70a1689f6ad33e921fe`.
- Rung-3 byte-exact comparator log SHA-256:
  `0ccc72b2c817ffc10f6d22764ed2625fc8c90ffc508ab283bce84e8263c80aca`.
- Proof program SHA-256:
  `17dd1de5965a938ccc1b11f4b7db6dd81aa3e2d02117804aaf8cfbd2960a6b0e`.

## Standing schema procedure

Future additive wire-schema changes stop the ladder for a ruling. If adopted,
their references are deliberately re-blessed once with an exact-delta receipt.
They are never hidden by automatic normalization.

The ladder resumes after rung 3 using executable source identical to
`881d1cbdfe`. The streaming-session suite, decode measurement, Spec V2 gate,
live-duel shape check, and lifecycle closure remain unspent.
