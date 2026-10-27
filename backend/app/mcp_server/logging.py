import logging


class MCPPrivacyFilter(logging.Filter):
    def filter(self, record):
        error_type = (
            record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else "none"
        )
        record.msg = "mcp_transport_event level=%s error_type=%s"
        record.args = (record.levelname, error_type)
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def configure_mcp_logging():
    for name in list(logging.Logger.manager.loggerDict):
        if name == "mcp" or name.startswith("mcp."):
            logger = logging.getLogger(name)
            if not any(isinstance(item, MCPPrivacyFilter) for item in logger.filters):
                logger.addFilter(MCPPrivacyFilter())
