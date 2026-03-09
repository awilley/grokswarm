"""
test_markdown.py -- 15 tests for a Markdown-to-HTML converter.

The converter should be implemented in markdown_converter.py
with a function: to_html(text: str) -> str

These tests define the expected behavior. Do NOT modify this file.
"""

from markdown_converter import to_html


class TestHeadings:
    def test_h1(self):
        assert to_html("# Hello") == "<h1>Hello</h1>"

    def test_h2(self):
        assert to_html("## Subheading") == "<h2>Subheading</h2>"

    def test_h3(self):
        assert to_html("### Third Level") == "<h3>Third Level</h3>"


class TestInlineFormatting:
    def test_bold(self):
        result = to_html("This is **bold** text")
        assert result == "<p>This is <strong>bold</strong> text</p>"

    def test_italic(self):
        result = to_html("This is *italic* text")
        assert result == "<p>This is <em>italic</em> text</p>"

    def test_inline_code(self):
        result = to_html("Use `print()` here")
        assert result == "<p>Use <code>print()</code> here</p>"


class TestCodeBlocks:
    def test_fenced_code_block(self):
        text = "```\nx = 1\ny = 2\n```"
        expected = "<pre><code>x = 1\ny = 2</code></pre>"
        assert to_html(text) == expected


class TestLinks:
    def test_link(self):
        result = to_html("[Click here](https://example.com)")
        assert result == '<p><a href="https://example.com">Click here</a></p>'


class TestLists:
    def test_unordered_list(self):
        text = "- Apple\n- Banana\n- Cherry"
        expected = "<ul><li>Apple</li><li>Banana</li><li>Cherry</li></ul>"
        assert to_html(text) == expected

    def test_ordered_list(self):
        text = "1. First\n2. Second\n3. Third"
        expected = "<ol><li>First</li><li>Second</li><li>Third</li></ol>"
        assert to_html(text) == expected


class TestParagraphs:
    def test_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        expected = "<p>First paragraph.</p>\n<p>Second paragraph.</p>"
        assert to_html(text) == expected


class TestEdgeCases:
    def test_escaped_asterisks(self):
        result = to_html("\\*not bold\\*")
        assert result == "<p>*not bold*</p>"

    def test_nested_formatting(self):
        result = to_html("**bold and *italic* inside**")
        assert result == "<p><strong>bold and <em>italic</em> inside</strong></p>"

    def test_empty_input(self):
        assert to_html("") == ""

    def test_mixed_content(self):
        text = "A paragraph with **bold** and [a link](http://x.com)."
        expected = '<p>A paragraph with <strong>bold</strong> and <a href="http://x.com">a link</a>.</p>'
        assert to_html(text) == expected
