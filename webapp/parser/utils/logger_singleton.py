from .shared_logger import SharedLogger, RichConsoleProxy

logger = SharedLogger()
console = RichConsoleProxy(logger=logger)

from .user_prompt import UserPrompt
prompt = UserPrompt()