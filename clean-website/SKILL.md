---
name: clean-website
description: Create or adapt a clean static research-project website using the bundled website template. Use when the user asks for a minimal project website, paper website, publication page, clean academic site, or a website using a clean or miniamlist template.
---

# Clean Website

Use this skill to create a standalone static website from the bundled template. The template follows a minimalist-style layout and contains: hero title/subtitle, placeholder authors, placeholder publication buttons, scroll hint, Abstract/Method/Results/BibTeX sections, three Results tabs, qualitative sub-toggles, quantitative placeholder tables, and footer attribution.

## Bundled Resources

- `assets/template/index.html`: standalone HTML template.
- `assets/template/style.css`: matching stylesheet.
- `scripts/copy_template.py`: helper that copies the template into a target folder.

## Workflow

1. Decide the output directory from the user request. If the user does not specify one, ask before writing files.
2. Copy the template:

```bash
python3 clean-website/scripts/copy_template.py /path/to/output
```

Use `--force` only when the user wants existing `index.html` or `style.css` overwritten.

3. Edit copied files in the target folder, not the skill assets, unless the user explicitly asks to update the skill itself.
4. Replace placeholders conservatively:
   - `Project Title`, `Project Subtitle`
   - `Author One` through `Author Four`
   - `Institution Name`
   - `abstract here`, `method here`, `results here`, `bibtex here`
   - placeholder publication buttons and table labels
5. Preserve the template attribution footer unless the user explicitly asks to change it.
6. Verify the result:
   - Confirm `index.html` only references available local files or intentional external links.
   - Parse or open `index.html` directly; no build step is required.
   - Check that Results tabs switch, qualitative sub-toggles open/collapse, and `Home` appears only after scrolling.

## Style Rules

- Keep the visual style aligned with `style.css`; avoid introducing unrelated design systems.
- Keep page sections simple and content-focused.
- Use linked text for URLs in visible content rather than displaying raw URLs, unless the user asks otherwise.
- Keep the footer link blue and underlined.
