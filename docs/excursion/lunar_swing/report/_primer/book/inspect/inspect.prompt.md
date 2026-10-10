Self-contained inspection bundle for *LUNAR-SWING 研究总结报告* (`lunar_swing_summary.pdf`, 66 pages).
The page images are listed below; paths are relative to this file (`inspect/`), rasterized at 110 dpi. Attach the images to the model in the listed order and ask it to judge them against the rubric.

# Visual quality-assurance of a typeset book PDF

You are reviewing rasterized pages of a Chinese book with CJK fonts and English citations
that `primer-book` converted from markdown to LaTeX (`ctexbook` class, xelatex) and
compiled to PDF. This is not a prose review: report concrete visible layout defects that a
programmer can fix in the markdown -> LaTeX converter, and only defects that the rendered
page alone can show.

## What counts as a defect

1. A table scaled or wrapped below legibility: cells colliding, single-character orphan
   lines, or a header drifting away from its rows.
2. A figure too small to read, blank, all-grey, or cropped at the frame.
3. A heading orphaned at the foot of a page, or stranded from its text by a page break.
4. A landscape page used where a portrait page would do, or a wide table left in portrait
   when it needed a landscape page.
5. Conspicuous whitespace: a large gap mid-page, a mostly empty page, content pushed onto
   a page of its own for no reason.
6. Anything else a reader would call wrong that only the rendered page can show.

## Verified elsewhere — do not report

Content crossing the text-block edge, chapter/figure/table numbering consistency, and
missing or garbled glyphs are already guaranteed by the deterministic `primer-book check
--deep` layer (the LaTeX log's `Missing character` line and the font-coverage check cover
the glyph class). Reason: that layer proves these at 100% recall and zero cost, so a
vision reply can add nothing — do not spend reasoning on them and do not report them.

## Converter elements `converter_hint` should name

- table layout plan (`tables.plan_table`: portrait fit / portrait wrap / portrait scale /
  landscape fit / landscape wrap, font ladder `-4` / `-5` / `-6`)
- table emitter (`tex.render_table`: xltabular for wrap, adjustbox for scale)
- figure placement (`tex.render_blocks`: includegraphics width=0.96\linewidth,
  height=0.68\textheight, caption/note handling)
- heading emitter (`tex.render_blocks`: \part / \chapter / \section)
- page geometry (`geometry` margin, `pdflscape` landscape pages)
- float placement and page breaks (`tex.render_blocks`: float specifiers, \clearpage)

Phrase the hint as an action for the programmer, for example: "wide table chose the
portrait-wrap tier but still overran; consider the landscape tier" or "figure scaled below
legibility although adjustbox was available".

## Output

Reply with JSON only, no prose around it:

{"findings": [{"page": <printed page number>, "code": "<kebab-case>",
  "severity": "error|warning|info", "what": "<the visible defect>",
  "where": "table|figure|listing|heading|page-geometry|glyph",
  "converter_hint": "<the markdown->TeX element most likely responsible, and what to change>"}]}

Use severity "error" only for defects that make the book wrong or unreadable. Return
{"findings": []} when the pages are clean. Suggested codes: table-illegible, figure-small,
figure-blank, figure-crop, heading-orphan, heading-stray, landscape-unnecessary,
landscape-missing, whitespace-gap, other.

## Pages in this request

| image | page | why this page |
|---|---|---|
| 1 (pages/page-0015.png) | 15 | explicitly selected with --pages |
| 2 (pages/page-0016.png) | 16 | explicitly selected with --pages |
| 3 (pages/page-0017.png) | 17 | explicitly selected with --pages |
| 4 (pages/page-0030.png) | 30 | explicitly selected with --pages |
| 5 (pages/page-0046.png) | 46 | explicitly selected with --pages |
| 6 (pages/page-0047.png) | 47 | explicitly selected with --pages |
