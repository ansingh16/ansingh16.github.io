---
title: 'Source Attribution Is the Point: A RAG Chatbot for UK Immigration Rules'
date: 2026-07-16
permalink: /posts/2026/07/uk-visa-rag-source-attribution/
tags:
  - rag
  - nlp
  - langchain
  - faiss
  - llm
---

Ask a general-purpose language model whether you qualify for a UK Skilled Worker visa and it will give you a fluent, plausible answer that you have no way to check. It might also be describing a rule that was repealed two years ago, in the same confident tone it uses for everything else.

On a subject where people make decisions based on the answer, that is a problem no amount of prompt tuning fixes. So I built a retrieval-augmented chatbot over the actual published Immigration Rules, designed so that every answer comes with the GOV.UK sections it was drawn from. The citation is the part I care about; the generated prose is a convenience on top of it.

This post walks through the corpus, the retrieval design, and the evaluation I have not done yet. The code is on [GitHub](https://github.com/ansingh16/UK_visa_RAG). Everything runs locally on a laptop --- no external API calls, no data leaving the machine.

## The corpus

The UK Immigration Rules are published on GOV.UK in 105 sections. I pull all of them through the GOV.UK Content API rather than scraping rendered pages, which means I get structured content with a stable URL per section instead of parsing whatever the page template does this month. BeautifulSoup cleans the raw HTML down to plain text.

The stable URL is what makes the rest possible. Every chunk carries the title and GOV.UK link of the section it came from, through retrieval and into the answer, so attribution is not something bolted on at the end.

## The pipeline

| Component | Choice |
|---|---|
| Extraction | GOV.UK Content API, BeautifulSoup, pandas |
| Chunking | Word-level, ~480 words with 60-word overlap |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS via LangChain |
| Orchestration | `ConversationalRetrievalChain` with a 10-turn memory window |
| Generation | Ollama, `mistral:instruct`, local |
| Retrieval | Top-5 chunks, each returned with content, title, URL and score |

The 60-word overlap is there because legal text carries conditions across paragraph boundaries constantly. A requirement and the exception to it routinely sit in adjacent paragraphs, and a clean split between them leaves a chunk that states the rule but drops the clause qualifying it. That chunk is worse than no chunk, because nothing about it looks incomplete. The overlap makes that less likely.

`all-MiniLM-L6-v2` is small and CPU-friendly, which is why the whole thing runs on a laptop. It is not the strongest embedding model going and I would expect a bigger one to retrieve better, but staying local was a constraint I set at the start and kept.

## Constraining the generator

The prompt does most of the safety work, and it is short:

- Use the provided context **only**.
- Ignore clauses marked "DELETED".
- Answer in plain English, bullet points when the rules are complex.
- Cite the relevant section titles at the end.

The "DELETED" line is the one that is doing real work. The Immigration Rules keep repealed clauses in the published text and mark them as deleted, so the corpus contains rules that are no longer rules. Semantic retrieval will surface them quite happily, because a deleted clause about a visa is still very much *about* that visa. I only found this by reading the source text, and it is the main reason I would tell anyone to read their corpus before designing a pipeline over it.

The chain returns source documents alongside the answer, and a debug mode prints the retrieved chunks with their scores. I use that constantly. Being able to see which sections produced an answer is most of what makes the thing trustworthy to me.

## Two implementations

There is a second, standalone implementation built on Haystack instead of LangChain, using dual retrieval: BM25 keyword matching alongside sentence-transformer embeddings, with a custom Ollama generator component.

The keyword half earns its place because immigration rules are full of exact identifiers: paragraph references, visa route names, specific numeric thresholds. Someone asking about "Appendix FM" wants that literal string, and dense embeddings are fairly indifferent to it where BM25 is not. Embeddings handle the intent, BM25 handles the identifiers, and this corpus throws up both kinds of question.

## What I have not measured

There is no quantitative evaluation in this project, and I would rather say so than let the writeup imply otherwise. No retrieval precision or recall at k, no answer-faithfulness scoring, no labelled question set. I have read a lot of outputs and they look reasonable to me, which is close to the weakest evidence available, since I am the person who built it and I already believe it works.

So what the repo currently demonstrates is a working local RAG pipeline with attribution wired end to end. It does not demonstrate that the retrieval is any good. Fixing that is not difficult, just unglamorous: write perhaps 50 questions where I know which section holds the answer, measure how often that section lands in the top 5, then check whether the cited sources actually support what the generated answer claims. That is the next thing I do on this project.

The citations are still earning their keep in the meantime. A wrong answer with a link attached can be checked in about ten seconds, which is a much lower bar than being right, but it is the bar that makes the tool usable while I still do not know how good the retrieval is.

## What I'd carry into the next one

The thing I keep coming back to is that the retrieval is the product here and the language model is mostly a presentation layer. A system that hands back the three most relevant sections and a link is genuinely useful even when its prose is clumsy. One that produces confident, well-written prose with no provenance is a liability, and being more fluent makes it slightly worse rather than better.

The DELETED clauses were the lesson I did not expect. Nothing in any RAG tutorial would have told me that this corpus contains rules that are no longer in force but still read like rules. That came from reading the text, and I suspect most domain-specific RAG traps are like that.

Keeping it local cost me some retrieval quality, and I would make the same call again --- immigration questions are personal enough that "nothing leaves the laptop" is a feature worth paying for.
