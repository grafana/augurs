#!/usr/bin/env python3
"""
Extract and test code examples from mdBook documentation.

This script parses markdown files, extracts code blocks within
langtabs sections, and runs them to ensure they're valid.
"""

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ANSI color codes for pretty output
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"


@dataclass
class CodeBlock:
    """Represents a code block extracted from markdown."""

    language: str
    code: str
    file: Path
    line: int
    attributes: list[str] | None = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = []


@dataclass
class TestResult:
    """Result of testing a code block."""

    success: Optional[bool]
    block: CodeBlock
    error: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None


def extract_code_blocks(file_path: Path) -> List[CodeBlock]:
    """Parse markdown file and extract code blocks within langtabs sections."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    in_lang_tabs = False
    current_language = None
    current_code = []
    current_start_line = 0

    lines = content.split("\n")

    current_attributes = []

    for i, line in enumerate(lines, start=1):
        if "<!-- langtabs-start -->" in line:
            in_lang_tabs = True
            continue

        if "<!-- langtabs-end -->" in line:
            # Save any remaining code block
            if current_language and current_code:
                blocks.append(
                    CodeBlock(
                        language=current_language,
                        code="\n".join(current_code),
                        file=file_path,
                        line=current_start_line,
                        attributes=current_attributes,
                    )
                )
                current_code = []
                current_language = None
                current_attributes = []
            in_lang_tabs = False
            continue

        if in_lang_tabs:
            # Check for code fence start with optional attributes
            fence_match = re.match(r"^```(\w+)(?:,(.+))?", line)
            if fence_match:
                # Save previous code block if exists
                if current_language and current_code:
                    blocks.append(
                        CodeBlock(
                            language=current_language,
                            code="\n".join(current_code),
                            file=file_path,
                            line=current_start_line,
                            attributes=current_attributes,
                        )
                    )
                current_language = fence_match.group(1)
                # Parse attributes if present
                if fence_match.group(2):
                    current_attributes = [
                        attr.strip() for attr in fence_match.group(2).split(",")
                    ]
                else:
                    current_attributes = []
                current_code = []
                current_start_line = i + 1
                continue

            # Check for code fence end
            if line.strip() == "```" and current_language:
                blocks.append(
                    CodeBlock(
                        language=current_language,
                        code="\n".join(current_code),
                        file=file_path,
                        line=current_start_line,
                        attributes=current_attributes,
                    )
                )
                current_code = []
                current_language = None
                current_attributes = []
                continue

            # Accumulate code lines
            if current_language:
                current_code.append(line)

    return blocks


def find_markdown_files(directory: Path) -> List[Path]:
    """Recursively find all markdown files in a directory."""
    return list(directory.rglob("*.md"))


def run_command(
    cmd: List[str], cwd: Optional[Path] = None, timeout: int = 30
) -> Dict[str, Any]:
    """Run a command and return stdout/stderr."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command timed out",
            "returncode": -1,
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def find_python_venv() -> Optional[str]:
    """Find the Python venv for pyaugurs if it exists."""
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    venv_python = repo_root / "crates" / "pyaugurs" / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return None


def setup_js_test_env(temp_dir: Path) -> Path:
    """Set up a package.json in the temp directory to use local augurs package."""
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    augurs_js_path = repo_root / "js" / "augurs"

    if augurs_js_path.exists():
        # Create a minimal package.json that references the local packages
        script_dir = Path(__file__).parent.resolve()
        repo_root = script_dir.parent.parent
        wasmstan_path = repo_root / "components" / "js" / "prophet-wasmstan"

        package_json = {
            "type": "module",
            "dependencies": {
                "@bsull/augurs": f"file:{augurs_js_path.absolute()}",
                "@bsull/augurs-prophet-wasmstan": f"file:{wasmstan_path.absolute()}"
                if wasmstan_path.exists()
                else "^0.2.0",
            },
        }
        package_json_path = temp_dir / "package.json"
        with open(package_json_path, "w") as f:
            json.dump(package_json, f, indent=2)

        # Run npm install to link the local package
        result = run_command(["npm", "install", "--silent"], cwd=temp_dir, timeout=60)
        if not result["success"]:
            print(
                f"  {Colors.YELLOW}Warning: Could not install local JS package: {result['stderr']}{Colors.RESET}"
            )
    else:
        print(
            f"  {Colors.YELLOW}Warning: augurs JS package not found at {augurs_js_path}{Colors.RESET}"
        )

    return temp_dir


def test_javascript(block: CodeBlock, temp_dir: Path, index: int) -> TestResult:
    """Test a JavaScript code block."""
    code = block.code

    # Skip if marked with no_test attribute
    if block.attributes and "no_test" in block.attributes:
        return TestResult(
            success=None, block=block, skipped=True, reason="Marked with no_test"
        )

    # Skip if it's just a comment or installation instruction
    stripped = code.strip()
    if stripped.startswith("//") and len(code.split("\n")) <= 3:
        return TestResult(
            success=None, block=block, skipped=True, reason="Installation/comment only"
        )

    # Skip npm install commands
    if "npm install" in code or "yarn add" in code:
        return TestResult(
            success=None, block=block, skipped=True, reason="Installation command"
        )

    file_path = temp_dir / f"test_{index}.js"

    # Detect which augurs modules are being used and add initSync calls
    # This matches how js/testpkg does it
    init_lines = []
    init_imports = []

    module_map = {
        "@bsull/augurs/mstl": ("mstl", "mstl_bg.wasm"),
        "@bsull/augurs/ets": ("ets", "ets_bg.wasm"),
        "@bsull/augurs/dtw": ("dtw", "dtw_bg.wasm"),
        "@bsull/augurs/outlier": ("outlier", "outlier_bg.wasm"),
        "@bsull/augurs/clustering": ("clustering", "clustering_bg.wasm"),
        "@bsull/augurs/transforms": ("transforms", "transforms_bg.wasm"),
        "@bsull/augurs/prophet": ("prophet", "prophet_bg.wasm"),
        "@bsull/augurs/seasons": ("seasons", "seasons_bg.wasm"),
        "@bsull/augurs/changepoint": ("changepoint", "changepoint_bg.wasm"),
    }

    # Check which modules are imported
    for module_path, (module_name, wasm_file) in module_map.items():
        if module_path in code:
            init_imports.append(
                f"import {{ initSync as initSync_{module_name} }} from '{module_path}';"
            )
            init_lines.append(
                f"initSync_{module_name}({{ module: readFileSync('node_modules/@bsull/augurs/{wasm_file}') }});"
            )

    # Prepend initialization code if needed
    if init_imports:
        init_code = "import { readFileSync } from 'node:fs';\n"
        init_code += "\n".join(init_imports) + "\n\n"
        init_code += "\n".join(init_lines) + "\n\n"
        code = init_code + code

    # Wrap in async function if needed for top-level await
    # BUT keep imports outside the async wrapper
    if "await " in code and "async function" not in code and "async ()" not in code:
        code_lines = code.split("\n")
        import_lines = []
        other_lines = []

        for line in code_lines:
            if line.strip().startswith("import "):
                import_lines.append(line)
            else:
                other_lines.append(line)

        # Build final code: imports first, then async wrapper
        final_lines = import_lines
        if import_lines:
            final_lines.append("")  # Blank line after imports
        final_lines.append("(async () => {")
        final_lines.extend(other_lines)
        final_lines.append("})();")

        code = "\n".join(final_lines)

    file_path.write_text(code, encoding="utf-8")

    result = run_command(["node", file_path.name], cwd=temp_dir)

    if result["success"]:
        return TestResult(success=True, block=block)
    else:
        return TestResult(
            success=False, block=block, error=result["stderr"] or result["stdout"]
        )


def test_python(
    block: CodeBlock, temp_dir: Path, index: int, python_cmd: str = "python3"
) -> TestResult:
    """Test a Python code block."""
    code = block.code

    # Skip if marked with no_test attribute
    if block.attributes and "no_test" in block.attributes:
        return TestResult(
            success=None, block=block, skipped=True, reason="Marked with no_test"
        )

    # Skip if it's just a comment or installation instruction
    stripped = code.strip()
    if stripped.startswith("#") and len(code.split("\n")) <= 3:
        return TestResult(
            success=None, block=block, skipped=True, reason="Installation/comment only"
        )

    # Skip pip install commands
    if "pip install" in code or "poetry add" in code:
        return TestResult(
            success=None, block=block, skipped=True, reason="Installation command"
        )

    file_path = temp_dir / f"test_{index}.py"
    file_path.write_text(code, encoding="utf-8")

    result = run_command([python_cmd, str(file_path)])

    if result["success"]:
        return TestResult(success=True, block=block)
    else:
        return TestResult(
            success=False, block=block, error=result["stderr"] or result["stdout"]
        )


def test_rust_batch(
    blocks: list[tuple[int, CodeBlock]], temp_dir: Path
) -> list[TestResult]:
    """Test all Rust code blocks together in a single Cargo project using modules."""
    if not blocks:
        return []

    # Create a single Cargo project
    project_dir = temp_dir / "rust_batch_tests"
    project_dir.mkdir(exist_ok=True)

    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)

    # Track which modules to create
    modules = []
    skipped_results = []

    for idx, (original_idx, block) in enumerate(blocks):
        code = block.code.strip()

        # Skip if marked with no_test attribute
        if block.attributes and "no_test" in block.attributes:
            skipped_results.append(
                (
                    original_idx,
                    TestResult(
                        success=None,
                        block=block,
                        skipped=True,
                        reason="Marked with no_test",
                    ),
                )
            )
            continue

        # Skip if it's just a comment or TOML config
        if (code.startswith("//") or "[dependencies]" in code) and len(
            code.split("\n")
        ) <= 5:
            skipped_results.append(
                (
                    original_idx,
                    TestResult(
                        success=None,
                        block=block,
                        skipped=True,
                        reason="Installation/comment only",
                    ),
                )
            )
            continue

        # Create a module file for this example
        module_name = f"test_{idx}"
        modules.append(module_name)
        module_file = src_dir / f"{module_name}.rs"

        # If code has fn main, rename it to run
        # Otherwise wrap in a run function
        if "fn main" in code:
            # Replace main with run
            code = code.replace("fn main()", "pub fn run()")
            code = code.replace("fn main() {", "pub fn run() {")
            code = code.replace("fn main() -> Result<", "pub fn run() -> Result<")
        else:
            # Wrap in run function
            code = f"#[allow(dead_code, unused_variables, unused_imports)]\npub fn run() {{\n{code}\n}}"

        # Add allow attributes at the top
        code = f"#![allow(dead_code, unused_variables, unused_imports)]\n\n{code}"

        module_file.write_text(code, encoding="utf-8")

    # Create main.rs that declares all modules and calls them
    main_code = ["// Auto-generated test runner", ""]
    for module_name in modules:
        main_code.append(f"mod {module_name};")
    main_code.append("")
    main_code.append("fn main() {")
    for module_name in modules:
        main_code.append(f"    {module_name}::run();")
    main_code.append("}")

    main_rs = src_dir / "main.rs"
    main_rs.write_text("\n".join(main_code), encoding="utf-8")

    # Create Cargo.toml
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    augurs_path = repo_root / "crates" / "augurs"

    cargo_toml = f"""[package]
name = "rust_batch_tests"
version = "0.1.0"
edition = "2021"

[dependencies]
augurs = {{ path = "{augurs_path}", features = ["mstl", "ets", "forecaster", "outlier", "clustering", "dtw", "seasons", "prophet", "prophet-wasmstan"] }}

[workspace]
"""

    cargo_toml_path = project_dir / "Cargo.toml"
    cargo_toml_path.write_text(cargo_toml, encoding="utf-8")

    # Run cargo check once for all tests
    result = run_command(["cargo", "check", "--quiet"], cwd=project_dir, timeout=120)

    results_list = []

    # If batch compilation succeeded, all tests pass!
    if result["success"]:
        for original_idx, block in blocks:
            skip_result = next(
                (r for i, r in skipped_results if i == original_idx), None
            )
            if skip_result:
                results_list.append((original_idx, skip_result))
            else:
                results_list.append(
                    (original_idx, TestResult(success=True, block=block))
                )
        return results_list

    # Batch compilation failed - test each module individually to get specific errors
    print(
        f"  {Colors.YELLOW}Batch compilation failed, testing individually...{Colors.RESET}"
    )

    module_idx = 0
    for original_idx, block in blocks:
        # Check if this was skipped
        skip_result = next((r for i, r in skipped_results if i == original_idx), None)
        if skip_result:
            results_list.append((original_idx, skip_result))
            continue

        module_name = f"test_{module_idx}"
        module_idx += 1

        # Create a temporary main.rs that only calls this one module
        temp_main = f"""// Test single module
mod {module_name};

fn main() {{
    {module_name}::run();
}}
"""
        temp_main_path = src_dir / "main.rs.tmp"
        temp_main_path.write_text(temp_main, encoding="utf-8")

        # Swap main.rs temporarily
        original_main = main_rs.read_text()
        main_rs.write_text(temp_main, encoding="utf-8")

        # Check just this module
        individual_result = run_command(
            ["cargo", "check", "--quiet"], cwd=project_dir, timeout=30
        )

        # Restore original main.rs
        main_rs.write_text(original_main, encoding="utf-8")

        if individual_result["success"]:
            results_list.append((original_idx, TestResult(success=True, block=block)))
        else:
            results_list.append(
                (
                    original_idx,
                    TestResult(
                        success=False,
                        block=block,
                        error=individual_result["stderr"]
                        or individual_result["stdout"],
                    ),
                )
            )

    return results_list


def test_code_block(
    block: CodeBlock,
    temp_dirs: Dict[str, Path],
    index: int,
    python_cmd: str = "python3",
) -> TestResult:
    """Test a single code block."""
    language = block.language.lower()

    if language in ("javascript", "js"):
        return test_javascript(block, temp_dirs["js"], index)
    elif language in ("python", "py"):
        return test_python(block, temp_dirs["python"], index, python_cmd)
    elif language in ("rust", "rs"):
        # Rust tests are batched, handled separately
        return None
    else:
        return TestResult(
            success=None,
            block=block,
            skipped=True,
            reason=f"Unsupported language: {language}",
        )


def main():
    """Main function."""
    print(f"{Colors.BLUE}Testing code examples in documentation{Colors.RESET}\n")

    script_dir = Path(__file__).parent.resolve()
    book_src_dir = script_dir.parent / "src"

    # Find Python venv
    python_venv = find_python_venv()
    if python_venv:
        print(f"{Colors.GREEN}Using Python venv:{Colors.RESET} {python_venv}\n")
        python_cmd = python_venv
    else:
        print(
            f"{Colors.YELLOW}Python venv not found, using system python3{Colors.RESET}\n"
        )
        python_cmd = "python3"

    # Use fixed directories for caching (not temp)
    # This allows Cargo to reuse builds across runs (much faster!)
    test_dir = script_dir.parent / ".test-examples"
    test_dir.mkdir(exist_ok=True)

    temp_dirs = {}
    for lang in ["js", "python", "rust"]:
        lang_dir = test_dir / lang
        lang_dir.mkdir(exist_ok=True)
        temp_dirs[lang] = lang_dir

    # Set up JS test environment with local package
    print(f"{Colors.BLUE}Setting up JavaScript test environment...{Colors.RESET}")
    setup_js_test_env(temp_dirs["js"])
    print()

    # Find all markdown files
    markdown_files = find_markdown_files(book_src_dir)
    print(f"Found {len(markdown_files)} markdown files\n")

    # Extract all code blocks
    all_blocks = []
    for md_file in markdown_files:
        blocks = extract_code_blocks(md_file)
        all_blocks.extend(blocks)

    print(f"Found {len(all_blocks)} code blocks in langtabs\n")

    # Separate Rust blocks from others for batch testing
    rust_blocks = []
    other_blocks = []

    for i, block in enumerate(all_blocks):
        if block.language.lower() in ("rust", "rs"):
            rust_blocks.append((i, block))
        else:
            other_blocks.append((i, block))

    # Test all Rust blocks together (much faster!)
    if rust_blocks:
        print(
            f"{Colors.BLUE}Testing {len(rust_blocks)} Rust examples in batch...{Colors.RESET}\n"
        )
        rust_results_list = test_rust_batch(rust_blocks, temp_dirs["rust"])
        rust_results_map = {idx: result for idx, result in rust_results_list}
    else:
        rust_results_map = {}

    # Now test all blocks in order, using batched Rust results where available
    results = []
    for i, block in enumerate(all_blocks):
        rel_path = block.file.relative_to(book_src_dir)
        print(
            f"{Colors.BLUE}Testing{Colors.RESET} {rel_path}:{block.line} ({block.language})"
        )

        # Get result (from batch for Rust, or test individually for others)
        if block.language.lower() in ("rust", "rs"):
            result = rust_results_map.get(i)
            if not result:
                # Shouldn't happen, but handle gracefully
                result = TestResult(
                    success=None, block=block, skipped=True, reason="Not in batch"
                )
        else:
            result = test_code_block(block, temp_dirs, i, python_cmd)

        if result.skipped:
            print(
                f"  {Colors.YELLOW}⊘ Skipped{Colors.RESET} {Colors.GRAY}({result.reason}){Colors.RESET}"
            )
        elif result.success:
            print(f"  {Colors.GREEN}✓ Passed{Colors.RESET}")
        else:
            print(f"  {Colors.RED}✗ Failed{Colors.RESET}")
            if result.error:
                # Print first few lines of error
                error_lines = result.error.strip().split("\n")[:5]
                for line in error_lines:
                    print(f"  {Colors.GRAY}{line}{Colors.RESET}")
                if len(result.error.strip().split("\n")) > 5:
                    print(f"  {Colors.GRAY}...{Colors.RESET}")

        results.append(result)
        print()  # Empty line between tests

    # Print summary
    passed = sum(1 for r in results if r.success is True)
    failed = sum(1 for r in results if r.success is False)
    skipped = sum(1 for r in results if r.skipped)

    print("─" * 60)
    print(f"{Colors.BLUE}Summary{Colors.RESET}")
    print(f"  {Colors.GREEN}Passed:{Colors.RESET}  {passed}")
    print(f"  {Colors.RED}Failed:{Colors.RESET}  {failed}")
    print(f"  {Colors.YELLOW}Skipped:{Colors.RESET} {skipped}")
    print(f"  Total:   {len(results)}")

    if failed > 0:
        print(f"\n{Colors.RED}Failed examples:{Colors.RESET}")
        for result in results:
            if result.success is False:
                rel_path = result.block.file.relative_to(book_src_dir)
                print(f"  - {rel_path}:{result.block.line} ({result.block.language})")

        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}All tests passed!{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
