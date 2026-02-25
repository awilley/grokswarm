import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
import shutil

async def main():
    session = PromptSession(erase_when_done=True)
    cols = shutil.get_terminal_size((80, 20)).columns
    line = '\u2500' * cols
    
    for i in range(3):
        msg = HTML(f'<style fg=\"#444444\">{line}</style>\n<b><ansibrightcyan>> </ansibrightcyan></b>')
        ans = await session.prompt_async(msg)
        print(f'> {ans}')
        print(f'Agent response {i}')

if __name__ == "__main__":
    asyncio.run(main())
