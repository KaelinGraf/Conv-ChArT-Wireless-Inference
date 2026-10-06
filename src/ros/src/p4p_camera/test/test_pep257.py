# Copyright 2025 Conv-ChArT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_pep257.main import main
import pytest

# ament_pep257's default ignore list omits D213, so it enforces "multi-line
# docstring summary should start at the SECOND line". D213 is the mutually
# exclusive twin of D212 ("...at the first line"), which is PEP 257's own
# recommendation and the style used throughout this repo: convchart_ros,
# p4p_serial_bridge and this package all put the summary on the first line.
# Both cannot hold at once, so D213 is ignored explicitly rather than left to
# fail 32 times -- commit 585a7c6 records the same unresolved failure for the
# serial bridge. The rest of ament's defaults are restated because --ignore
# replaces the list rather than adding to it.
IGNORE = [
    'D100', 'D101', 'D102', 'D103', 'D104', 'D105', 'D106', 'D107',
    'D203', 'D212', 'D213', 'D404',
]


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    rc = main(argv=['.', 'test', '--ignore', *IGNORE])
    assert rc == 0, 'Found code style errors / warnings'
