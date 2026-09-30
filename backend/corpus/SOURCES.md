# Knowledge-base sources (v2)

PDFs in this folder are **not committed** (`*.pdf` is gitignored; CLAUDE.md: never commit raw
downloaded data). This file records where each one came from and its SHA-256, so the exact
source of every fact record can be re-obtained and checked.

Choosing sources is a HUMAN task (CLAUDE.md). The status column records the owner's decision;
"candidate" means not yet decided.

| File | What it is | Origin | SHA-256 | Status |
|---|---|---|---|---|
| `UASB_POP_2025_official_full.pdf` | UAS Bengaluru, *Package of Practices 2025* (Kannada, Nudi-encoded). Sugarcane = ch. 34, PDF pages 173–187 (printed 165–179); pest/disease table printed pp. 174–176 | https://www.uasbangalore.edu.in/wp-content/uploads/2026/06/POP-Final-Book-2025.pdf (downloaded 2026-09-30) | `0e636dcde59df4325aef9ab20edecf2bbd14dc37c3bc29fe7c0aab64bbf92ad7` | **primary, South Karnataka** (owner decision 2026-09-30) |
| `POP-Final-Book-2025.pdf` | Owner's copy of the same book, printed pp. 65–176; the owner deliberately removed pp. 177–179 (jaggery making, out of scope). Pages 165–176 are text-identical to the official copy | owner download | — | same content; **v2 scope = printed pp. 165–176** |
| `UAS Package of Practices (POP) 2023 – … GKVK, BENGALURU – 560065.pdf` | UAS Bengaluru POP 2023, sugarcane chapter only (15 pp., Nudi). Same content as 2025 with one page inserted in 2025 | owner download (mirror: rawe2020.in) | `66f545bacea629706d5d20500a431addb39a5151ca246bb8299c33e00d104092` | superseded by 2025 |
| `PoP Agriculture cover page2020.cdr.pdf` | Sugarcane chapter (9 pp., Nudi, printed p. 225 on) of a 2020 Karnataka POP; mentions Dharwad/Belagavi, 2017-18 statistics; text differs from UAS-B. Publisher confirmed by owner 2026-09-30: **UAS Dharwad** | owner download | `908538f854e8e9eb1ccf24aa1ec47fc89a7e8ec02d84cfc41a332239df7b2597` | **primary, North Karnataka** |
| ICAR Kharif Agro-Advisories for Farmers 2025, multi-language edition, p. 133 (Karnataka, Kannada) | National advisory; Karnataka sugarcane section in Unicode Kannada. Differs from UAS-B on P, K and Trichogramma rate; one paragraph (rhizome weevil, thrips, propiconazole) looks like it belongs to another crop | https://icar.org.in/sites/default/files/Circulars/ICAR%20Kharif%20Agro-Advisories%20for%20Farmers%202025__multi%20language%2B%2BNew.pdf | not downloaded yet | candidate, cross-check only |
| `Sugarcane.pdf` | NIPHM, *AESA based IPM Package: Sugarcane* (English, 66 pp., 2014) | https://niphm.gov.in/IPMPackages/Sugarcane.pdf | `0c1bfe634185444da43ff217c9910eed6dd1a5a356437f81cd16c0668a672609` | cross-check only (2014; not Karnataka-specific) |
| `07_Sugarcane.pdf` | TNAU Crop Production Guide, ch. 7 Sugarcane (English, Tamil Nadu) | owner download | `2915f7e1b36338c487eaa998880c3ab19bbcb76b2d9a81b171be8f3ffd0634f5` | cross-check only (Tamil Nadu) |
| `…SP_Sugarcane2017.pdf` | Status Paper on Sugarcane 2017 (national, all states; likely Directorate of Sugarcane Development) | owner download | `e92fa9905cd73650fdd29cf416b6e4ba3fdfebb436c1fcae5f5565d90e0d9cef` | report background only, not a dose source |
| `crop-practice-sugarcane.pdf` | Nigerian extension leaflet (2016) recommending Endosulfan (banned in India since 2011) and Nuvacron | owner download (deleted 2026-09-30) | — | **REJECTED — unsafe, never index** |
| `sugarcane-crop-plan-fertigation.pdf` | Fertilizer company product plan ("Krista") | owner download (deleted 2026-09-30) | — | **REJECTED — commercial, never index** |

## Conflict rule (owner decision 2026-09-30)
When UAS and ICAR (or other cross-check sources) disagree, the bot gives the **UAS**
recommendation and may note that ICAR's national advisory differs. Cross-check sources never
supply a dose on their own. UAS Bengaluru vs UAS Dharwad differences are regional, not
conflicts: each record carries its region, and the bot must not blend the two.

## Converting the Nudi PDFs
The UAS books use the legacy Nudi font encoding, so extracted text looks like `PÀ§Äâ` (= ಕಬ್ಬು).
Converted with `knconverter` from https://github.com/aravindavk/ascii2unicode (GPL-3.0, commit
`6d53b9b`), used as an external tool and not vendored. Checked on its own test file and on known
words from the chapter. Table cells must be read in PDF stream order (`sort=False`), because
position-sorted extraction scrambles Nudi vowel signs. Numbers and units convert unchanged.
Cells in the pest table are merged across pests, so column→pest mapping is not reliable: every
extracted record is a DRAFT until a person confirms it against the printed page.
