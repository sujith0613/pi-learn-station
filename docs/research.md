# Research: Dyslexic Letter Confusions

Evidence behind the confusion set in `ml/confusions.py`. This drives the whole
design: the app's purpose is to catch and gently correct these specific,
well-documented confusions using sentence context.

## Mirror / reversible letters — Tier A

The Latin (Roman) alphabet contains **exactly four lowercase letters that
differ solely by orientation** (mirror images): **b, d, p, q**. Because they
differ only by orientation contrast, they are the classic "reversal" letters.

- **Fernandes & Leite (2017)** — *Mirrors are hard to break* (PubMed 28285044):
  dyslexic children fail to **automatize mirror discrimination** during visual
  object processing; their shape-based judgments are immune to mirror-image
  differences. Mirror-generalization (processing b and d as equivalent) persists
  and is specific to mirror images, not plane rotations.
- **Perea et al. (2011)** — *Suppression of mirror generalization for reversible
  letters*: reading systems actively suppress mirror images of **reversible**
  letters (b/d) but not non-reversible ones (c, r). Reversible letters are
  exactly b↔d and p↔q.
- **University of Lisbon (2025)** — *Mirror invariance dies hard during letter
  processing by dyslexic college students*: even dyslexic **adults** show
  residual mirror invariance for reversible letters in masked priming. This is
  not something children just "grow out of" — it needs support.
- **Martínez et al. (same-different task, Spanish)** — reversible letters
  (b,d,p,q) vs non-reversible (g,h,m,v): all children are slower/more error-prone
  on reversible letters; **vertical-axis** (up-down: b–p, d–q) mirror errors are
  more frequent than horizontal-axis (left-right: b–d, p–q) for reversible
  letters. → Include **both axes** in the pair set.
- **Terepocki et al. (2002)** — *Incidence and nature of letter orientation
  errors*: reading-disabled children make more orientation confusions than
  average readers on reversible items.

### Directional asymmetry (important for priors)
- **Treiman & Kessler (2011) / PMC 4309997** — children reverse **left-facing**
  letters far more than right-facing ones. Left-facing: `a d g j q y z`;
  right-facing: `b c e f h k n p r s`. So expect more **d→b** and **q→p** errors
  than the reverse. The char-LM absorbs this via word likelihood.

## Phonological voicing pairs — Tier B

When children spell by sound, the most frequent phonological-based spelling
errors are the **voiceless/voiced cognate pairs**:

- **PLOS One (2019)** — *Auditory temporal training & voiceless/voiced-based
  orthographic errors*: "one of the most frequent phonological-based
  orthographic errors is related to voiceless/voiced phonemes." Uses exactly
  the pairs /p/-/b/, /t/-/d/, /k/-/g/, /f/-/v/, /s/-/z/, /ch/-/j/ in
  phonologically balanced dictation. (Cross-linguistic: Brazilian Portuguese,
  Italian, English.)
- **Bahr, Silliman & Berninger (2012, JSLHR)** — analysis of 888 writers'
  spelling: phonological errors (incl. voicing substitutions) are a persistent
  error class across grades.
- **Friðriksdóttir & Ingason (2020, ACL READI)** — confirmation that **context
  disambiguation of confusion sets** is the right tool for dyslexic spelling
  support; ~73-86% accuracy with a decision-tree classifier on a confusion-set
  corpus. Validates our char-LM-as-context-brain architecture.
- **Rello et al. (2015)** — ~20% of dyslexic misspellings are **real-word
  errors** (e.g., "when" for "than") that only context can resolve. This is why
  the transformer scores candidates against the surrounding sentence.

## Stretch — Tier C (flag off by default)
- **Friedmann et al. — letter position dyslexia**: "slime"→"smile",
  "warp"→"wrap", "form/from"; migrations of letters within a word, worst in
  middle positions. Could be flagged by whole-word masking (same mechanism).
- **Rello & Llisterri (2012)** — vowel substitutions follow phonetic features
  (a↔e, i↔e, o↔e); weak signal for a char-LM at our scale — skipped.

## Sources
- Fernandes & Leite 2017, *Mirrors are hard to break*, PubMed 28285044
- Perea et al. 2011, *Suppression of mirror generalization for reversible
  letters*, J. Exp. Psych: Learning, Memory & Cognition
- Univ. of Lisbon 2025, *Mirror invariance dies hard during letter processing*
- Martínez et al., *Same-different letter decision task* (Spanish children)
- Terepocki, Kruk & Willows 2002, *Incidence and nature of letter orientation
  errors*, J. Learning Disabilities
- Treiman & Kessler 2011, *Statistical learning, letter reversals, and reading*
  (PMC 4309997)
- PLOS One 2019, *Voiceless/voiced-based orthographic errors*
- Bahr, Silliman & Berninger 2012, *Linguistic pattern analysis of
  misspellings*, JSLHR
- Friðriksdóttir & Ingason 2020, *Disambiguating confusion sets as an aid for
  dyslexic spelling*, ACL READI workshop
- Rello, Ballesteros & Bigham 2015, *Dyslexia and real-word spelling errors*