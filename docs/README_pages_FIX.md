---
layout: default
title: "GitHub Pages Fix Notes"
---

# GitHub Pages Mermaid Rendering Fix

This document describes changes made to restore proper mermaid diagram rendering
in the GitHub Pages site after script-regenerated `docs/*.md` files stopped
rendering mermaid diagrams.

## Root Causes

1. **Missing YAML front matter** - Regenerated markdown files were missing the
   required front matter block that Jekyll uses to process files.
2. **Code block mismatch** - Jekyll renders ` ```mermaid ` fenced blocks as
   `<pre><code class="language-mermaid">` but Mermaid.js expects
   `<div class="mermaid">` elements.
3. **Asset path issues** - Scripts and styles need `relative_url` filter to work
   correctly with the site's `baseurl`.

## Changes Made

### 1. YAML Front Matter

Added front matter to `pipeline_map.md` and `project_audit.md` (the files
containing mermaid blocks that were missing it):

```yaml
---
layout: default
title: "Page Title"
---
```

### 2. Layout Updates (`_layouts/default.html`)

- Added `defer` attribute to all script tags for better loading performance
- Split Mermaid initialization into dedicated `mermaid-init.js`
- Ensured all asset paths use `relative_url` filter

### 3. New `assets/js/mermaid-init.js`

This script:
- Waits for DOMContentLoaded
- Converts Jekyll's `<pre><code class="language-mermaid">` blocks to
  `<div class="mermaid">` elements that Mermaid.js expects
- Initializes Mermaid with dark theme configuration
- Includes retry logic with 5-second max timeout for async CDN loading

### 4. Refactored `assets/js/custom.js`

- Removed duplicate Mermaid initialization code (now handled by `mermaid-init.js`)
- Kept loading animation and other UI enhancements

### 5. Configuration (`_config.yml`)

Ensured correct `url` and `baseurl` settings:
```yaml
url: "https://basic-nature.github.io"
baseurl: "/html_Parser_prototype"
```

## For Future Generators

When regenerating `docs/*.md` files, ensure each file has YAML front matter:

```yaml
---
layout: default
title: "Your Page Title"
---
```

The `layout: default` line is required for Jekyll to process the file and apply
the site template. The `title` is optional but recommended.

## Verification

To verify the fix works locally:

```bash
# Build the site
bundle exec jekyll build --source docs --destination _site

# Check generated HTML contains mermaid divs
grep -r "class=\"mermaid\"" _site/pipeline_map.html
grep -r "class=\"mermaid\"" _site/project_audit.html

# Serve locally to test
bundle exec jekyll serve --source docs
```

In browser DevTools:
- Network tab: Ensure no 404s for `mermaid-init.js` or `mermaid.min.js`
- Console: Check for no mermaid-related errors
- Page: Verify diagrams render correctly
