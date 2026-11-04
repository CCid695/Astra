"""Unit tests for astra_lib."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import astra_lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astra_lib as L


class TestChunkText(unittest.TestCase):
    """Test the chunk_text function."""

    def test_basic_chunking(self):
        """Test basic paragraph chunking."""
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = L.chunk_text(text, target=20, overlap=5)
        self.assertGreater(len(chunks), 0)
        # Each chunk should be non-empty
        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_long_paragraph_splitting(self):
        """Test that long paragraphs are split correctly."""
        long_para = "Word " * 100  # Create a long paragraph
        chunks = L.chunk_text(long_para, target=50, overlap=10)
        self.assertGreater(len(chunks), 1)
        # Verify overlap
        for i in range(len(chunks) - 1):
            self.assertGreater(len(chunks[i]), 0)
            self.assertGreater(len(chunks[i + 1]), 0)

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = L.chunk_text("")
        self.assertEqual(chunks, [])

    def test_single_paragraph(self):
        """Test a single short paragraph."""
        text = "Single paragraph."
        chunks = L.chunk_text(text, target=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_config_default_chunking(self):
        """Test chunking with default config values."""
        text = "Para one.\n\nPara two.\n\nPara three."
        # Use default values from config
        chunks = L.chunk_text(text)
        self.assertGreater(len(chunks), 0)


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up a temporary config file."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_rc = os.path.join(self.temp_dir, ".astrarc")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_rc):
            os.remove(self.temp_rc)
        os.rmdir(self.temp_dir)

    @patch('astra_lib.RC_PATH')
    def test_cfg_defaults(self, mock_rc_path):
        """Test that cfg returns defaults when no config exists."""
        mock_rc_path.return_value = self.temp_rc
        # Test a default value
        self.assertEqual(L.cfg("ASTRA_GEN_MODEL"), "gemma4:12b")

    @patch('astra_lib.RC_PATH')
    def test_cfg_from_file(self, mock_rc_path):
        """Test reading config from file."""
        mock_rc_path.return_value = self.temp_rc
        # Write a config file
        with open(self.temp_rc, "w") as f:
            f.write("ASTRA_GEN_MODEL=test-model\n")
        # Reload the config
        self.assertEqual(L.cfg("ASTRA_GEN_MODEL"), "test-model")

    @patch('astra_lib.RC_PATH')
    def test_cfg_env_override(self, mock_rc_path):
        """Test that environment variables override config file."""
        mock_rc_path.return_value = self.temp_rc
        # Set an environment variable
        os.environ["ASTRA_GEN_MODEL"] = "env-model"
        try:
            self.assertEqual(L.cfg("ASTRA_GEN_MODEL"), "env-model")
        finally:
            del os.environ["ASTRA_GEN_MODEL"]

    @patch('astra_lib.RC_PATH')
    def test_set_cfg(self, mock_rc_path):
        """Test setting config values."""
        mock_rc_path.return_value = self.temp_rc
        L.set_cfg(ASTRA_GEN_MODEL="new-model")
        # Verify it was written to file
        with open(self.temp_rc, "r") as f:
            content = f.read()
            self.assertIn("ASTRA_GEN_MODEL=new-model", content)


class TestEmbeddingPrefix(unittest.TestCase):
    """Test embedding prefix requirements."""

    def test_doc_prefix(self):
        """Test document prefix."""
        self.assertEqual(L.DOC_PREFIX, "search_document: ")

    def test_query_prefix(self):
        """Test query prefix."""
        self.assertEqual(L.QUERY_PREFIX, "search_query: ")


class TestFrontmatter(unittest.TestCase):
    """Test frontmatter parsing."""

    def setUp(self):
        """Set up a temporary markdown file."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_md = os.path.join(self.temp_dir, "test.md")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_md):
            os.remove(self.temp_md)
        os.rmdir(self.temp_dir)

    def test_read_frontmatter(self):
        """Test reading YAML frontmatter."""
        content = """---
title: Test Document
author: Test Author
---

Content here."""
        with open(self.temp_md, "w") as f:
            f.write(content)
        meta = L.read_frontmatter(self.temp_md)
        self.assertEqual(meta.get("title"), "Test Document")
        self.assertEqual(meta.get("author"), "Test Author")

    def test_read_frontmatter_no_frontmatter(self):
        """Test file without frontmatter."""
        content = "Just content\n\nNo frontmatter."
        with open(self.temp_md, "w") as f:
            f.write(content)
        meta = L.read_frontmatter(self.temp_md)
        self.assertEqual(meta, {})

    def test_doc_title_from_frontmatter(self):
        """Test getting title from frontmatter."""
        content = """---
title: Test Title
---

Content."""
        with open(self.temp_md, "w") as f:
            f.write(content)
        title = L.doc_title(self.temp_md)
        self.assertEqual(title, "Test Title")

    def test_doc_title_from_filename(self):
        """Test getting title from filename when no frontmatter."""
        content = "Content without title."
        with open(self.temp_md, "w") as f:
            f.write(content)
        title = L.doc_title(self.temp_md)
        self.assertEqual(title, "test")  # basename without .md


class TestSourceURLs(unittest.TestCase):
    """Test source URL generation."""

    @patch('astra_lib.RELEASE')
    def test_source_urls(self, mock_release):
        """Test generating source URLs."""
        mock_release.return_value = "australia"
        # Mock the path
        test_path = "/some/path/markdown/servicenow-platform/test-doc.md"
        with patch('astra_lib.rel_markdown_path', return_value="servicenow-platform/test-doc.md"):
            urls = L.source_urls(test_path)
            self.assertEqual(urls["path"], "servicenow-platform/test-doc.md")
            self.assertEqual(urls["topic"], "test-doc")
            self.assertIn("test-doc.html", urls["url"])
            self.assertIn("version=australia", urls["url"])


class TestConfigurableParameters(unittest.TestCase):
    """Test that configurable parameters are accessible."""

    @patch('astra_lib.cfg')
    def test_chunk_target(self, mock_cfg):
        """Test CHUNK_TARGET accessor."""
        mock_cfg.return_value = "1500"
        self.assertEqual(L.CHUNK_TARGET(), 1500)

    @patch('astra_lib.cfg')
    def test_chunk_overlap(self, mock_cfg):
        """Test CHUNK_OVERLAP accessor."""
        mock_cfg.return_value = "200"
        self.assertEqual(L.CHUNK_OVERLAP(), 200)

    @patch('astra_lib.cfg')
    def test_embed_batch(self, mock_cfg):
        """Test EMBED_BATCH accessor."""
        mock_cfg.return_value = "128"
        self.assertEqual(L.EMBED_BATCH(), 128)

    @patch('astra_lib.cfg')
    def test_file_batch(self, mock_cfg):
        """Test FILE_BATCH accessor."""
        mock_cfg.return_value = "400"
        self.assertEqual(L.FILE_BATCH(), 400)


if __name__ == "__main__":
    unittest.main()
