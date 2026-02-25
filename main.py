"""GrokSwarm entry point — thin shim that imports the package and runs the CLI."""

from grokswarm.shared import app
import grokswarm.repl       # registers @app.callback and chat command
import grokswarm.commands   # registers swarm/team/task/expert commands
import grokswarm.dashboard  # registers dashboard command

if __name__ == "__main__":
    app()
