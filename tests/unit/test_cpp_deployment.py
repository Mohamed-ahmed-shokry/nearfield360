from __future__ import annotations

from pathlib import Path

from nearfield360.config import InferenceConfig


def test_cpp_deployment_files_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    deploy_dir = repo_root / "deploy" / "cpp"

    assert deploy_dir.is_dir()
    assert (deploy_dir / "CMakeLists.txt").is_file()
    assert (deploy_dir / "README.md").is_file()
    assert (deploy_dir / "src" / "main.cpp").is_file()

    inc_dir = deploy_dir / "include" / "nearfield360"
    assert (inc_dir / "tensorrt_engine.hpp").is_file()
    assert (inc_dir / "fisheye_preprocessor.hpp").is_file()
    assert (inc_dir / "perception_pipeline.hpp").is_file()


def test_cpp_deployment_header_integrity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inc_dir = repo_root / "deploy" / "cpp" / "include" / "nearfield360"

    for header in inc_dir.glob("*.hpp"):
        content = header.read_text(encoding="utf-8")
        assert "#pragma once" in content
        assert "namespace nearfield360 {" in content


def test_cpp_preprocessor_matches_inference_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    header = repo_root / "deploy" / "cpp" / "include" / "nearfield360" / "fisheye_preprocessor.hpp"
    content = header.read_text(encoding="utf-8")

    cfg = InferenceConfig()
    # Check that default normalization in C++ matches Python configuration
    assert str(cfg.mean[0]) in content
    assert str(cfg.mean[1]) in content
    assert str(cfg.mean[2]) in content
    assert str(cfg.std[0]) in content
    assert str(cfg.std[1]) in content
    assert str(cfg.std[2]) in content
