"""
Tests that the repo's .gitignore keeps calibration sessions out of git.

calibrate.py writes each session (frames/*.png, calibration_log.yaml, the config backup) to
<output_dir>/<YYYYmmdd-HHMMSS>/, and output_dir defaults to <config dir>/../calibration_output,
which for the documented -p config:=<repo>/cfg/cfg.yaml is the repo root. The repo's .gitignore
has to cover that folder, or every session shows up in `git status` and `git add -A` commits its
frames. Skipped without git or outside a git checkout of the repo.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

SESSION = '20261005-044153'             # a session folder name, <YYYYmmdd-HHMMSS>
SESSION_FILES = ['frames/view_000.png', 'frames/rejected_000.png', 'calibration_log.yaml',
                 'cfg.yaml.bak']


def find_repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'cfg' / 'cfg.yaml').is_file() and (parent / '.git').exists():
            return parent
    return None


@pytest.mark.parametrize('name', SESSION_FILES)
def test_default_session_folder_is_ignored_by_the_repo_gitignore(name):
    repo = find_repo()
    git = shutil.which('git')
    if repo is None or git is None:
        pytest.skip('needs git and a git checkout of the repo')
    config = repo / 'cfg' / 'cfg.yaml'
    path = config.parent.parent / 'calibration_output' / SESSION / name    # default output_dir
    # One line per path, <source>:<line>:<pattern><TAB><path>, with empty fields when no rule
    # matches; --no-index leaves the decision to the ignore rules alone.
    result = subprocess.run([git, '-C', str(repo), 'check-ignore', '--no-index', '--verbose',
                             '--non-matching', str(path)], capture_output=True, text=True)
    assert result.returncode in (0, 1), result.stderr
    source, _, pattern = result.stdout.partition('\t')[0].split(':', 2)
    assert source == '.gitignore' and not pattern.startswith('!'), \
        f'.gitignore does not ignore {path.relative_to(repo)} (git: {result.stdout.strip()})'
