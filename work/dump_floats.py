import importlib.util
import sys

sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("mc", "work/manuscript_content.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

lines = ["=== FIGURES ==="]
for key, value in mc.FIGURES.items():
    lines.append("--- %s ---" % key)
    lines.append(repr(value))

lines.append("")
lines.append("=== TABLES ===")
for key, value in mc.TABLES.items():
    lines.append("--- %s ---" % key)
    lines.append(repr(value))

open("work/current_floats.txt", "w", encoding="utf-8").write("\n".join(lines))
print(len(lines))
