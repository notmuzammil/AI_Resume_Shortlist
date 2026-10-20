# Scores, Progress Bars, Experience Comparisons and Export Schema

## What a score means

The original algorithm in `modules/scoring.py` is unchanged. It calculates **JD fit**, not estimated performance or hiring probability. Default component weights are skills 45%, relevant experience 32%, education-field fit 13%, and role-title alignment 10%; components absent from a JD can be disabled and other weights renormalized. Requirements have priority and relevance weights. Strong inferences can earn partial confidence; related skills are not equal to direct matches.

A final-score ceiling may apply for missing must-haves, missing title-identified skills, or insufficient relevant professional experience. Bachelor's degree and graduates-only rules can reject applicants. Accordingly, *the final match percentage is not the simple average of visible bars*.

## Visual progress bars

- **Overall match**: `candidate.score` numeric 0–100, produced by the scoring engine.
- **Skills match**: `candidate.components.skills.percent` numeric 0–100, weighted requirements before caps; separate **coverage count %** is the fraction of JD requirements with any evidence and is not the weighted score.
- **Each JD requirement**: `candidate.requirements[].match_percent` numeric 0–100. 100 represents full requirement coverage, lower percentages reflect accepted but lower confidence/partial credit; 0 represents no supporting CV evidence located. Evidence and match type accompany every bar.
- **Relevant experience score**: `candidate.components.experience.percent` is the actual weighted scoring component. Its detail shows JD minimum and counted relevant years.
- **Relevant experience comparison**: capped `candidate.relevant_experience.percent` = `min(100, relevant_years / JD explicit minimum * 100)`; if no explicit minimum, the UI labels its 3-year reference as an **informational display benchmark**. This is a separate explanatory bar, not the scoring component.
- **Total experience comparison**: capped `candidate.total_experience.percent` = `min(100, total_years / reference_years * 100)`. This is **never used as an independently weighted hiring score**.
- **Education/title**: weighted component percent or `null` if the JD did not allow scoring that component.
- **Non-scored status**: final score is JSON `null`, never a fabricated zero-point match.

## JSON structure

A report is a plain JSON object, `schema_version:1`, with these primary fields:

```json
{
  "schema_version": 1,
  "job_id": "UUID",
  "job_title": "Senior Backend Engineer",
  "generated_at": "UTC ISO 8601",
  "duration_seconds": 1.1,
  "jd": {
    "title": "Senior Backend Engineer",
    "required_years": 3.0,
    "requirements": [{"skill": "nodejs", "importance": "must", "source": "..."}]
  },
  "candidates": [{
    "candidate_id": "UUID",
    "filename": "resume.pdf",
    "rank": 1,
    "status": "scored",
    "category": "Highly Suitable",
    "score": 91.5,
    "total_years": 5.0,
    "relevant_years": 4.0,
    "components": {
      "skills": {"percent": 95.0, "available": true, "detail": "..."},
      "experience": {"percent": 88.0, "available": true, "detail": "..."}
    },
    "requirements": [{
      "skill": "nodejs", "importance": "must", "match_percent": 100.0,
      "matched": true, "match_type": "direct", "hands_on": true,
      "confidence": "", "evidence": "Built Node.js services ..."
    }],
    "relevant_experience": {"years": 4.0, "reference_years": 3.0, "percent": 100.0},
    "total_experience": {"years": 5.0, "reference_years": 3.0, "percent": 100.0},
    "flags": [], "notes": []
  }],
  "statistics": {"processed": 1, "scored": 1, "average_score": 91.5},
  "suggestions": ["Verify qualifications and hands-on experience before decisions."]
}
```

**The numbers above are illustrative, not a live scored case.** Exact reports contain additional fields such as evidence basis, work roles, unmatched must-haves, warnings, skill coverage and complete category statistics. CSV is flattened for spreadsheets, with dangerous formula-style prefixes neutralized. Raw full CV text is *not* included in the JSON exports, though individual quoted evidence may contain personal information and must be handled as confidential.

## Data-quality warnings

PDF OCR can misrecognize numbers, timelines or skill names. Skill inference does not prove hands-on work. Role relevance and durations rely on extracted dates, and incomplete degrees should not fulfill completed-degree requirements. All key claims require human confirmation before hiring decisions.
