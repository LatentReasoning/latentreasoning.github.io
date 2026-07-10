---
title: "A small, durable home for research notes"
description: "A Markdown-first blog template with the calm, spacious reading experience of Distill."
date: 2026-07-10
authors:
  - name: "Ada Lovelace$^1$"
    url: https://example.com
  - name: "Alan Turing$^2$"
affiliations:
  - "$^1$ Analytical Engine Lab"
  - "$^2$ Computing Machinery Group"
tags: [announcement, template]
bibliography: references.bib
citation_style: apa
acknowledgements: >
  We thank the NLCo group for testing the draft template and reporting
  small typography issues.
---

Research writing deserves a calm reading surface. This template keeps publishing deliberately small: every article is a Markdown file with a short YAML header, and the build produces only static files.

The practical conventions of scientific writing are well served by TeX [@knuth1984texbook; @lamport1994latex].

## Start with the idea

Write the argument in prose. Use ordinary Markdown for lists, links, code, and images. The rendered page uses a generous serif reading column, with a sans-serif title and byline inspired by the visual rhythm of [Distill](https://distill.pub/).

> A useful template should disappear once the writing begins.

### Equations use LaTeX

Inline mathematics works with dollar delimiters: the familiar identity $E = mc^2$ needs no special Markdown syntax.

The same source also supports display equations:

$$
\operatorname{softmax}(z)_i =
\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.
$$

### Charts use Chart.js

Use a `chart` code fence containing standard Chart.js configuration JSON:

```chart
{
  "type": "line",
  "data": {
    "labels": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
    "datasets": [
      {
        "label": "Validation score",
        "data": [0.41, 0.55, 0.63, 0.71, 0.78],
        "borderColor": "#004276",
        "backgroundColor": "rgba(0, 66, 118, 0.12)",
        "fill": true,
        "tension": 0.28
      }
    ]
  },
  "options": {
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "title": { "display": true, "text": "A compact Chart.js example" }
    },
    "scales": {
      "y": { "beginAtZero": true, "max": 1 }
    }
  }
}
```

### Tables are ordinary Markdown

| Rendering method | Source form         | Best for                  |
| ---------------- | ------------------- | ------------------------- |
| MathJax          | LaTeX delimiters    | Equations and notation    |
| Chart.js         | `chart` JSON fence  | Interactive data graphics |
| Markdown table   | Pipe-delimited rows | Compact comparisons       |

### Code blocks preserve source

Python is rendered in a fenced code block:

```python
def temperature_scale(logits, temperature=0.7):
    return [logit / temperature for logit in logits]
```

### A figure can be wide

Place an image in the post as usual. To make it extend beyond the text column, wrap it in a `figure` element and add `class="l-page"`:

<figure class="l-page">
  <img src="images/placeholder.svg" alt="A simple blue and green abstract landscape">
  <figcaption>Wide media has room to breathe without changing the reading measure.</figcaption>
</figure>

## Publishing is just a build

Run `npm install` once, then `npm run build`. GitHub Actions builds the `dist/` folder and deploys it to GitHub Pages. No Ruby, Jekyll, or server runtime is involved.

## Citation

```bibtex
@article{attention2017,
  author = {Vaswani, Ashish and others},
  title = {Attention Is All You Need},
  year = {2017},
  journal = {Advances in Neural Information Processing Systems}
}
```
