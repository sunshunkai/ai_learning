from __future__ import annotations

import unittest
from pathlib import Path

from tools.skill_loader import discover_skills, upload_files


class RepositorySkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skill_root = Path(__file__).resolve().parents[1] / "skills"
        self.skills = discover_skills(self.skill_root)

    def test_repository_exposes_progressive_loading_catalog(self) -> None:
        self.assertEqual(
            {skill.name for skill in self.skills},
            {
                "incident-triage",
                "json-normalizer",
                "python-project-starter",
                "release-notes",
                "text-reversal",
            },
        )
        for skill in self.skills:
            self.assertTrue(skill.description)
            self.assertTrue(skill.metadata.get("triggers"))
            self.assertTrue(skill.metadata.get("allowed_roles"))
            self.assertTrue(skill.metadata.get("version"))

    def test_standard_yaml_parses_all_frontmatter(self) -> None:
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")

        for skill in self.skills:
            text = skill.path.read_text(encoding="utf-8")
            frontmatter = text.split("---", 2)[1]
            parsed = yaml.safe_load(frontmatter)
            self.assertEqual(parsed["name"], skill.name)

    def test_incident_skill_upload_includes_referenced_checklist(self) -> None:
        incident = next(
            skill for skill in self.skills if skill.name == "incident-triage"
        )

        filenames = {filename for filename, _, _ in upload_files(incident)}

        self.assertIn("SKILL.md", filenames)
        self.assertIn("references/checklist.md", filenames)


if __name__ == "__main__":
    unittest.main()
