"""Vendored subset of the MGP Python common layer.

Ported from ``clotohub-servers/servers/common/`` so this server runs
standalone without depending on that (now private) monorepo:

- :mod:`_vendored_mcp_common.validation` — graceful-degradation argument
  validators (``validate_bool`` / ``validate_str`` / ``validate_int`` /
  ``validate_dict`` / ``validate_float`` / ``validate_list``).
- :mod:`_vendored_mcp_common.mcp_utils` — ``ToolRegistry`` MCP tool
  registration helper.

Only the symbols this server actually imports are vendored; keep this
copy in sync with the upstream common layer when it changes.
"""
