"""chemistree's MCP surface: one shared tool set, two servers.

``tools.py`` holds the tool set and its docstrings — the text an agent reads to
understand chemistree — registered against a backend. ``server.py`` is the
standalone, in-process server (``chemistree-mcp``) that any agent can add with
``claude mcp add``; the web app registers the same tools against an HTTP backend.
"""
