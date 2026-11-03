# Registration of the S(M,T) research on the Bitcoin blockchain

Two proofs sit in this folder. Each is an OpenTimestamps file: it ties the SHA256 hash of a document to a Bitcoin block, so anyone can check that the document existed by that block's date. No third party is trusted. Nothing but the hash leaves this folder.

| Document | SHA256 | Proof | Bitcoin block |
|---|---|---|---|
| The S(M,T) roadmap paper of July 2026 (the PDF stamped on 13 July 2026; kept in our records) | df4cc4b42e0c5e13fc148ccbe8e42de7cc67e5aa5416090595e205d3aaa9dd39 | SMT_ROADMAP_2026-07-13.pdf.ots | 957741 and 957754 |
| SMT_Router_Paper_v1.4.pdf, the public paper in this repository | c9baaebe69691129ff60c097a6304c34d2943225d9d9b31f74445298930e0b4d | SMT_Router_Paper_v1.4.pdf.ots | pending on 3 November 2026; upgraded once the calendar's Bitcoin transaction confirms |

## How to check, without installing anything

1. Open https://opentimestamps.org.
2. Drag the proof file (the .ots) and the document into the page.
3. The page says which Bitcoin block attests the document and the date of that block.

## How to check from a terminal

```
pip install opentimestamps-client
ots upgrade SMT_Router_Paper_v1.4.pdf.ots      # fetch the completed proof from the calendar
ots info SMT_Router_Paper_v1.4.pdf.ots         # shows the Bitcoin block height
ots verify SMT_Router_Paper_v1.4.pdf.ots       # full check, needs a Bitcoin node
```

Registered by P0P (PnotP), Pranav Lakherwal. Method: OpenTimestamps, first used on 13 July 2026.
