import importlib.util
import sys

sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("mc", "work/manuscript_content.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

lines = ["=== ABSTRACT ==="]
for heading, text in mc.ABSTRACT:
    lines.append("[%s] %s" % (heading, text))
words = len(" ".join(text for _, text in mc.ABSTRACT).split())
lines.append("(%d words)" % words)

for kind, payload in mc.BODY:
    if isinstance(payload, (list, tuple)):
        lines.append("## %s" % kind)
        for item in payload:
            lines.append("   - %s" % (item,))
    else:
        lines.append("<%s> %s" % (kind, payload))

open("work/current_body.txt", "w", encoding="utf-8").write("\n".join(lines))
print(len(lines))
