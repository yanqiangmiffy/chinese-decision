import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.convert_gguf import checkout_revision, main


class ConversionTests(unittest.TestCase):
    def test_source_distribution_git_pointer_is_optional_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / ".git").write_text("gitdir: ../../unavailable.git")
            with patch("scripts.convert_gguf.subprocess.check_output",
                       side_effect=subprocess.CalledProcessError(128, ["git"])):
                self.assertIsNone(checkout_revision(path))

    def test_2b_export_uses_identity_and_keeps_calibration_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, checkout = root / "renamed-release", root / "llama"
            base.mkdir()
            checkout.mkdir()
            (base / "model.safetensors").write_bytes(b"source weights")
            config = {"model_name": "chinese-decision-2b", "temperature": 1.07487}
            (base / "decision_config.json").write_text(json.dumps(config))
            (base / "training_summary.json").write_text(json.dumps({"selected_checkpoint": {"step": 4994}}))
            (checkout / "convert_hf_to_gguf.py").write_text("# test converter")

            def convert(command, **kwargs):
                Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF" + b"fixture")

            with patch("scripts.convert_gguf.subprocess.run", side_effect=convert), patch("builtins.print"):
                main(["--base", str(base), "--llama-cpp", str(checkout)])
            out = root / "chinese-decision-2b-gguf"
            self.assertTrue((out / "chinese-decision-2b-Q8_0.gguf").is_file())
            self.assertTrue((out / "chinese-decision-2b-BF16.gguf").is_file())
            self.assertFalse((root / "chinese-decision-0.8b-gguf").exists())
            self.assertEqual(json.loads((out / "decision_config.json").read_text()), config)
            self.assertEqual(json.loads((out / "training_summary.json").read_text())["selected_checkpoint"]["step"], 4994)

            # A second run must stop before invoking the converter or replacing metadata.
            with patch("scripts.convert_gguf.subprocess.run") as runner, self.assertRaises(FileExistsError):
                main(["--base", str(base), "--llama-cpp", str(checkout)])
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
