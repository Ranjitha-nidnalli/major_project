# Source documents for the next three crops (paddy, ragi, tomato)

Status as of 2026-11-25. "Opened" means the page or PDF was fetched and read in
this check; "not opened" means it only appeared in search results and has not
been verified.

## Finding

The UAS Bengaluru files that are easy to find online are **annual lists of
new technologies recommended for inclusion in the Package of Practices**. They
are addenda, not full cultivation guides. They contain some useful records
(for example tomato leaf curl virus, paddy gall midge doses) but they do not
replace the full book. The full Package of Practices for paddy, ragi and
tomato has **not yet been located**.

The sugarcane records cite printed page numbers (e.g. page 165), so they
come from a full book. The same kind of book is needed for each new crop.

## Opened

| Document | Issuer | Year | Covers | Use |
| --- | --- | --- | --- | --- |
| [New technologies recommended for inclusion in Package of Practices](https://www.uasbangalore.edu.in/wp-content/uploads/2025/06/technologies-E-2025.pdf) (file "technologies-E-2025") | UAS Bengaluru (issuer inferred from the text, not stated on the page) | 2024-25 file; year not stated in the text | 26 technologies. Paddy: 125% NPK trial, gall midge and hispa doses, rodent bait. Ragi: drone spraying. Tomato: leaf curl virus programme, imidacloprid and thiamethoxam doses | Supplement only. Not a full crop guide |
| [UAS Bengaluru "Varieties and technologies" page](https://www.uasbangalore.edu.in/en/varieties-and-technologies/) | UAS Bengaluru | 2018-19 to 2025-26 | Links to yearly technology lists in English and Kannada | Index of the addenda |

## Not opened (search results only)

- [Technologies developed 2000-2025](https://www.uasbangalore.edu.in/wp-content/uploads/2026/05/technologies-developed-2000-2025.pdf), UAS Bengaluru. May collect the addenda in one file.
- [Good agricultural practices in tomato cultivation: a technical manual for Karnataka](https://snrd-asia.org/wp-content/uploads/2024/07/27-Good-agricultural-practices-in-Tomato-Cultivation-–-A-technical-manual-for-Karnataka.pdf). The fetch failed. Issuer and year unverified.
- [Essentiality of Package of Practices of tomato cultivation in semi-arid regions](https://www.isec.ac.in/wp-content/uploads/2023/07/WP-525-Govindappa-final-1.pdf) (ISEC working paper). A study about tomato practices, not a package itself.

## To do (needs a person)

1. Obtain the full UAS Bengaluru *Package of Practices for Agricultural Crops* (paddy, ragi) in English or Kannada. A friend at UAS Mandya, whose college falls under UAS Bengaluru, can likely get the current edition or its PDF.
2. Obtain the horticultural crops package (tomato) from UHS Bagalkot or the UAS Bengaluru horticulture volume. Confirm which one applies to the target districts.
3. For each book record: title, issuer, edition year, language, and which districts it covers. Put these in the record `source` and `region` fields, as for sugarcane.
4. Do not build records from the addenda alone. Use them only to cross-check a dose against the full book.
5. Check pesticide names against the CIB&RC registration list before a record is confirmed (see `registration_helper.py`).
