import hashlib

import pytest

from midi_lm.datasets import verify_download_hash


def test_verify_download_hash(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
    expected_hash = hashlib.md5(b"This is a test file.").hexdigest()
    verify_download_hash(file_path, expected_hash)


def test_verify_download_hash_with_wrong_hash(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
    wrong_hash = hashlib.md5(b"This is not the same content.").hexdigest()

    with pytest.raises(AssertionError, match="Hash mismatch:"):
        # intentionally provide the wrong hash to check to ensure it raises an error
        verify_download_hash(file_path, wrong_hash)


def test_verify_download_hash_with_different_hash_algo(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
    expected_hash = hashlib.sha256(b"This is a test file.").hexdigest()
    verify_download_hash(file_path, expected_hash, hash_algo="sha256")
