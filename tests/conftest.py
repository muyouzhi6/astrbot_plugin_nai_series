"""Test this checkout even when its directory has a publish/staging suffix."""

import sys
import types
from pathlib import Path

package = types.ModuleType("astrbot_plugin_nai_series")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
