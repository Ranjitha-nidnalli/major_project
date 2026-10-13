# Literature survey: candidate papers

Found by web search on 2026-10-13. **Nothing here has been read yet.** The one-line notes
come from search-result abstracts and may be incomplete or wrong. Before a paper goes into
Chapter 2 or `references.bib`:

1. Open the link and read at least the abstract, method and results.
2. Take the performance numbers from the paper itself, not from this list.
3. Get the BibTeX from the official page (arXiv "Export BibTeX", ACL Anthology "Cite",
   publisher page). Do not type author lists from memory.
4. Tick the box and write one line in your own words: what they did, how well it worked,
   and the limitation that matters to us.

The guide asks for 20–25 papers. 24 are listed. Drop any that turn out to be weak (for
example a poster abstract with no results).

## A. Agricultural chatbots and QA (closest to our work)

| ✓ | Paper | Where | Why it matters to us |
|---|---|---|---|
| [ ] | *Intent Aware Context Retrieval for Multi-Turn Agricultural Question Answering* ("Krishi Sathi") | [arXiv 2508.03719](https://arxiv.org/abs/2508.03719) | Indian agri RAG chatbot; multi-turn slot collection before retrieval |
| [ ] | *Farmer.Chat: Scaling AI-Powered Agricultural Services for Smallholder Farmers* | [arXiv 2409.08916](https://arxiv.org/abs/2409.08916) | Deployed RAG advisory (Digital Green); scale and field use |
| [ ] | *AgroLLM: Connecting Farmers and Agricultural Practices through LLMs…* | [arXiv 2503.04788](https://arxiv.org/abs/2503.04788) | RAG agri chatbot for knowledge transfer |
| [ ] | *Empowering farmers with AI: a RAG-based LLM advisory framework* | [J. Agricultural Engineering, art. 1908](https://www.agroengineering.org/jae/article/view/1908) | RAG advisory on cultivation, pests, fertiliser |
| [ ] | *AgriLLM: Harnessing Transformers for Farmer Queries* | [arXiv 2407.04721](https://arxiv.org/abs/2407.04721) | Uses Kisan Call Centre queries |
| [ ] | *KisanQRS: A Deep Learning-based Automated Query-Response System for Agricultural Decision-Making* | [arXiv 2411.08883](https://arxiv.org/abs/2411.08883) | KCC call logs; reports F1 per state |
| [ ] | *Agribot: agriculture-specific question answer system* | [arXiv 2509.21535](https://arxiv.org/abs/2509.21535) | KCC-based chatbot: weather, market, plant protection |
| [ ] | *AgAsk: An Agent to Help Answer Farmer's Questions From Scientific Documents* | [arXiv 2212.10762](https://arxiv.org/abs/2212.10762) | Retrieval over scientific documents for farmers |
| [ ] | *GPT-4 as an Agronomist Assistant? Answering Agriculture Exams Using LLMs* | [arXiv 2310.06225](https://arxiv.org/abs/2310.06225) | LLM accuracy on agri exams incl. India; RAG used |
| [ ] | *GPT-4 as Evaluator: Evaluating LLMs on Pest Management in Agriculture* | [arXiv 2403.11858](https://arxiv.org/abs/2403.11858) | Pest-advice quality; LLM-as-judge (compare with our judge failure) |
| [ ] | *Farmer's Friend: IoT Fusion using Generative AI for farmer chatbot in Kannada* | [NeurIPS 2023 workshop page](https://neurips.cc/virtual/2023/84071) | Kannada farmer chatbot (check if a full paper exists) |
| [ ] | *AgriTalk: Revolutionizing Farming with a Multilingual Chatbot* | [Amrita publication page](https://www.amrita.edu/publication/agritalk-revolutionizing-farming-with-a-multilingual-chatbot/) | Multilingual incl. Kannada (find the actual venue) |

## B. Indic-language QA and evaluation

| ✓ | Paper | Where | Why it matters to us |
|---|---|---|---|
| [ ] | *Indic QA Benchmark: A Multilingual Benchmark to Evaluate QA Capability of LLMs for Indic Languages* | [arXiv 2407.13522](https://arxiv.org/abs/2407.13522) | Context-grounded QA in 11 Indian languages; weak low-resource performance |
| [ ] | *L3Cube-IndicQuest* | [arXiv 2409.08706](https://arxiv.org/abs/2409.08706) | QA benchmark incl. Kannada |
| [ ] | *IndicGenBench* | [arXiv 2404.16816](https://arxiv.org/abs/2404.16816) | Generation benchmark, 29 Indic languages |
| [ ] | *IndicMT Eval: A Dataset to Meta-Evaluate MT Metrics for Indian Languages* (Sai et al., ACL 2023) | [arXiv 2212.10180](https://arxiv.org/abs/2212.10180) | n-gram metrics like BLEU correlate worse with humans on Indian languages; supports our metric finding |

## C. Retrieval, RAG and embeddings

| ✓ | Paper | Where | Why it matters to us |
|---|---|---|---|
| [ ] | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 | [arXiv 2005.11401](https://arxiv.org/abs/2005.11401) | RAG foundation |
| [ ] | *M3-Embedding (BGE-M3): Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* | [arXiv 2402.03216](https://arxiv.org/abs/2402.03216) | Our embedding model (dense + sparse) |
| [x] | Cormack, Clarke, Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009 | [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Our fusion method (k = 60). In `references.bib` |
| [ ] | Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* | [arXiv 2309.15217](https://arxiv.org/abs/2309.15217) | LLM-based RAG evaluation; contrast with our finding that a faithfulness judge misses wrong-entity answers |

## D. Hallucination, abstention, metrics

| ✓ | Paper | Where | Why it matters to us |
|---|---|---|---|
| [ ] | Huang et al., *A Survey on Hallucination in LLMs: Principles, Taxonomy, Challenges, and Open Questions*, ACM TOIS 2024 | [arXiv 2311.05232](https://arxiv.org/abs/2311.05232) | Why fluent wrong answers happen |
| [ ] | *Know Your Limits: A Survey of Abstention in Large Language Models*, TACL 2025 | [arXiv 2407.18418](https://arxiv.org/abs/2407.18418) | Abstention as a safety tool (our core idea) |
| [x] | Kamath, Jia, Liang, *Selective Question Answering under Domain Shift*, ACL 2020 | [ACL Anthology](https://aclanthology.org/2020.acl-main.503) | Answer-or-abstain evaluation; model confidence is unreliable out of domain |
| [x] | Popović, *chrF: character n-gram F-score for automatic MT evaluation*, WMT 2015 | [ACL Anthology W15-3049](https://aclanthology.org/W15-3049) | Metric we use (and show its limits) |

`[x]` means the citation details were checked on the official page and the entry is in
`references.bib`. It still has to be read before it goes into the survey table.

## Foundational references (cited in the text, not survey entries)
BLEU (Papineni et al., ACL 2002), ROUGE (Lin, 2004), BM25 (Robertson and Zaragoza, 2009),
rule of three (Hanley and Lippman-Hand, JAMA 1983). Get the BibTeX from the official pages.
