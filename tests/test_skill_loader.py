from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.skill_loader import (
    SkillLoadError,
    discover_skills,
    load_skill_resource,
    upload_files,
)


class SkillLoaderMetadataTests(unittest.TestCase):
    def test_discover_skills_preserves_frontmatter_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "incident-triage"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: incident-triage
description: Triage application errors.
triggers: error, exception, stack trace
allowed_roles: developer, operator
version: 1.0.0
---

# Incident Triage
""",
                encoding="utf-8",
            )

            skill = discover_skills(temp_dir)[0]

            self.assertEqual(
                skill.metadata["triggers"],
                "error, exception, stack trace",
            )
            self.assertEqual(skill.metadata["allowed_roles"], "developer, operator")
            self.assertEqual(skill.metadata["version"], "1.0.0")

    def test_load_skill_resource_stays_inside_skill_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "incident-triage"
            references = skill_dir / "references"
            references.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: incident-triage
description: Triage application errors.
---

# Incident Triage
""",
                encoding="utf-8",
            )
            (references / "checklist.md").write_text(
                "# Checklist\n",
                encoding="utf-8",
            )

            skill = discover_skills(temp_dir)[0]

            self.assertEqual(
                load_skill_resource(skill, "references/checklist.md"),
                "# Checklist\n",
            )
            with self.assertRaises(SkillLoadError):
                load_skill_resource(skill, "../../outside.txt")

    def test_upload_files_includes_resources_with_relative_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "incident-triage"
            references = skill_dir / "references"
            references.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: incident-triage
description: Triage application errors.
---

# Incident Triage
""",
                encoding="utf-8",
            )
            (references / "checklist.md").write_text(
                "# Checklist\n",
                encoding="utf-8",
            )

            files = upload_files(discover_skills(temp_dir)[0])

            self.assertEqual(
                [filename for filename, _, _ in files],
                ["SKILL.md", "references/checklist.md"],
            )


if __name__ == "__main__":
    unittest.main()
