import json
import zipfile
from pathlib import Path

import pytest

from tcmdiscern.datasets import clinical_text, load_records, normalized_input


def test_read_official_jsonl_with_json_extension(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    source.write_text('{"norm_syndrome":"甲"}\n{"norm_syndrome":"乙"}\n', encoding="utf-8")
    assert len(load_records(source)) == 2


def test_read_json_array(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.json"
    source.write_text(json.dumps([{"Name": "甲"}]), encoding="utf-8")
    assert len(load_records(source)) == 1


def test_reject_non_record_json(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL"):
        load_records(source)


def test_clinical_fields_exclude_diagnosis_and_gold() -> None:
    record = {
        "chief_complaint": "頭痛",
        "description": "近兩日發熱",
        "detection": "脈浮",
        "norm_syndrome": "敏感金標",
        "syndrome": "另一個金標",
        "lcd_name": "疾病金標",
    }
    text = clinical_text(record)
    assert "頭痛" in text and "脈浮" in text
    assert all(word not in text for word in ("敏感金標", "另一個金標", "疾病金標"))
    assert "脈浮" not in clinical_text(record, include_four_examinations=False)
    assert normalized_input(record, full=True).startswith("chief_complaint:")


def test_zip_import_extracts_only_expected_files(tmp_path: Path) -> None:
    from scripts.fetch_tcmsd import unpack_selected

    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name in ("train.json", "dev.json", "syndrome_knowledge.json", "syndrome_vocab.txt"):
            output.writestr("nested/" + name, "test\n")
        output.writestr("../../outside.txt", "should not extract")
    unpack_selected(archive, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "train.json",
        "dev.json",
        "syndrome_knowledge.json",
        "syndrome_vocab.txt",
    }
    assert not (tmp_path / "outside.txt").exists()
