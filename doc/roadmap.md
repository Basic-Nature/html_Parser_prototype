# docs/roadmap.md

# Smart Elections Parser Roadmap

This roadmap outlines the key milestones and features planned for the Smart Elections Parser project.

---

## ✅ MVP: Functional Prototype (Completed)
- [x] HTML-based race scanning
- [x] State router and format router with fallback logic
- [x] Manual CAPTCHA handling with headless/browser toggle
- [x] Input cycling from `urls.txt`
- [x] `.env` support for configurable behavior
- [x] Output organized by state → county → race
- [x] Shared logic for election year/race interpretation

---

## 🔜 Next Milestone: Usability & Contribution
- [ ] CLI argument support (`--headless`, `--auto`, etc.)
- [ ] Refactor all output to support summary rollups
- [ ] Logging toggles for DEBUG/INFO modes
- [ ] Auto-detect known Enhanced Voting patterns
- [ ] Visual dashboard for extracted data validation (basic Flask UI?)
- [ ] End-to-end test harness with mock HTML snapshots

---

## 📦 Modularization Goals
- [ ] Fully encapsulate `utils/` for importable tools
- [ ] Separate `input_mode` for pre-saved local files
- [ ] Convert more handler logic into `handlers/shared/`
- [ ] Normalize vendor formats (Scytl, Hart, Clear Ballot)

---

## 🧠 Machine Learning Assistants (Long Term)
- [ ] ML-assisted candidate/party normalization
- [ ] NLP-based table structure inference
- [ ] Fuzzy precinct reconciliation across uploads

---

## 🙋 Community Involvement
- [ ] Add more `handlers/` for underrepresented counties
- [ ] Expand support for 2023 and 2021 elections
- [ ] Develop accessibility-first dashboards
- [ ] Translate output summaries to Spanish

---

## 💡 Inspiration
- Election Integrity Partnership (EIP)
- MIT Election Data & Science Lab
- Verified Voting tools

---

## 📫 Interested in helping?
See `CONTRIBUTING.md` or reach out via the issues page on GitHub.

---

