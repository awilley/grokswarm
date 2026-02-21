"""Fix mojibake in main.py - direct pattern replacement for corrupted UTF-8."""

filepath = r"c:\Users\aaron\grokswarm\main.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# These are the exact mojibake patterns left after the first partial fix.
# Original UTF-8 bytes were misread as cp1252, then some chars got replaced.
direct_replacements = [
    # em-dash: E2 80 94 -> a(E2) euro(80) "(94->already replaced to ")
    ("\u00e2\u20ac\"", "--"),
    # box horizontal: E2 94 80 -> a(E2) "(94->"") euro(80)
    ("\u00e2\"\u20ac", "-"),
    # box corner (bottom-left): E2 94 94 -> a ""
    ("\u00e2\"\"", "+"),
    # box tee (left): E2 94 9C -> a " oe
    ("\u00e2\"\u0153", "+"),
    # box vertical: E2 94 82 -> a " lowquote
    ("\u00e2\"\u201a", "|"),
    # right arrow: E2 86 92 -> a dagger '(already replaced)
    ("\u00e2\u2020'", "->"),
    # cross mark: E2 9D 8C -> a 9D OE
    ("\u00e2\x9d\u0152", "[FAIL]"),
    # prompt/angle: E2 9D AF -> a 9D macron
    ("\u00e2\x9d\u00af", ">"),
    # left triangle: E2 97 80 -> a --(97 was em-dash, replaced to --) euro
    ("\u00e2--\u20ac", "<"),
    # dots/ellipsis: E2 80 A6 -> a euro ...
    ("\u00e2\u20ac\u2026", "..."),
]

print("Direct pattern replacements:")
for old, new in direct_replacements:
    count = content.count(old)
    if count:
        print(f"  {repr(old)} -> {repr(new)}: {count}")
        content = content.replace(old, new)

# Final check
remaining = []
for i, ch in enumerate(content):
    if ord(ch) > 127:
        remaining.append((i, ch, f"U+{ord(ch):04X}"))
if remaining:
    print(f"\nWARNING: {len(remaining)} non-ASCII chars remaining:")
    # Show unique chars
    unique = set((ch, code) for _, ch, code in remaining)
    print(f"  Unique: {unique}")
    # Show first few in context
    for pos, ch, code in remaining[:5]:
        start = max(0, pos - 20)
        end = min(len(content), pos + 20)
        ctx = content[start:end].replace("\n", "\\n")
        print(f"  pos {pos}: {code} in: ...{ctx}...")
else:
    print("\nAll clean - pure ASCII!")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("File rewritten.")
