================================================================================
LIDA Multi-Agent Deliberation - Quick Start Guide
================================================================================

This guide shows how to run multi-agent deliberation experiments.

PREREQUISITES
-------------
- Python 3.8+
- Redis server installed
- Dependencies installed: pip install -r requirements.txt
- OpenRouter API key set:

    # Option 1: Load from file (recommended)
    export OPENROUTER_API_KEY=$(cat openrouter.txt)

    # Option 2: Set directly
    export OPENROUTER_API_KEY="sk-or-v1-..."


================================================================================
STEP 1: START THE SERVICES
================================================================================

Start Redis and the LIDA API server:

    SWARM_LIVE=true ./run.sh 6400 2400 start

Where:
    6400 = Redis port
    2400 = API port (use any available port)
    SWARM_LIVE=true = Enable live LLM calls (otherwise runs in mock mode)

You should see output like:
    Starting Redis on port 6400...
    Starting API server on port 2400...
    Services started successfully.


================================================================================
STEP 2: CHECK SERVICE STATUS
================================================================================

Check if services are running:

    ./run.sh 6400 2400 status

Or check the API directly:

    curl -s "http://localhost:2400/health"

Expected output:
    {"status": "healthy", ...}


================================================================================
STEP 3: RUN A DELIBERATION
================================================================================

Option A: Using a scenario file
-------------------------------

    python3 run_deliberation.py --port 2400 --scenario basic \
        --topic "Will AGI arrive before the year 2030?"

Option B: Specifying agents directly
------------------------------------

    python3 run_deliberation.py --port 2400 \
        --topic "Should AI development be open source?" \
        --agent "Yann LeCun" \
        --agent "Sam Altman" \
        --agent "Geoffrey Hinton"

Common arguments:
    --port PORT         API port (required, must match run.sh)
    --scenario NAME     Scenario file from scenarios/ folder
    --topic "TEXT"      The question for agents to deliberate
    --agent "NAME"      Add specific agent (can repeat, overrides scenario)
    --timeout SECONDS   Max wait time (0 = infinite, default)
    --poll-interval N   Seconds between status checks (default: 5)

Available agent names (examples):
    "Yann LeCun", "Sam Altman", "Geoffrey Hinton", "Yoshua Bengio",
    "Elon Musk", "Dario Amodei", "Ilya Sutskever", "Andrew Ng",
    "Stuart Russell", "Nick Bostrom", "Gary Marcus", "Andrej Karpathy"

Shorter aliases also work: LeCun, Altman, Hinton, Bengio, Musk, etc.


================================================================================
STEP 4: CHECK DELIBERATION STATUS
================================================================================

While a deliberation is running, check its status:

    curl -s "http://localhost:2400/api/deliberation/status" | jq

Example output:
    {
      "active": true,
      "phase": "voting",
      "topic": "Will AGI arrive before the year 2030?",
      "total_messages": 15,
      "consensus": {
        "Support": 2,
        "Oppose": 1,
        "Modify": 0,
        "Abstain": 0,
        "Continue": 0
      }
    }

Other useful API endpoints:

    # List all deliberations
    curl -s "http://localhost:2400/api/deliberations" | jq

    # Get LLM call logs
    curl -s "http://localhost:2400/api/llm/logs" | jq

    # Get server stats
    curl -s "http://localhost:2400/api/stats" | jq


================================================================================
STEP 5: WEB BROWSER ACCESS
================================================================================

Open the web UI in your browser:

    http://localhost:2400

The web interface shows:
    - Live deliberation progress
    - Agent messages and votes
    - Consensus tracking
    - LLM logs viewer

You can also start deliberations directly from the web UI.


================================================================================
STEP 6: STOP SERVICES
================================================================================

When finished, stop the services:

    ./run.sh 6400 2400 stop

Or stop just Redis:

    ./run.sh 6400 stop


================================================================================
VIEWING LOGS
================================================================================

Deliberation logs are saved to the logs/ folder:

    logs/deliberation_<scenario>_<id>_<timestamp>.log      - Status log
    logs/deliberation_<scenario>_<id>_<timestamp>.llm_logs.json - LLM API logs

Option A: Interactive terminal viewer (recommended)
---------------------------------------------------

    python3 view_logs.py logs/deliberation_basic_*.llm_logs.json

Controls:
    Up/Down, j/k  - Navigate entries
    Enter/Space   - View full entry details
    q/Esc         - Back / Quit
    /             - Search
    n             - Next search result
    Home/g        - Go to top
    End/G         - Go to bottom

Option B: Web browser log viewer
--------------------------------

    http://localhost:2400/log-viewer.html

Option C: Command line
----------------------

    # View status log
    cat logs/deliberation_basic_*.log

    # View LLM logs (pretty print)
    cat logs/deliberation_basic_*.llm_logs.json | jq


================================================================================
RUNNING MULTIPLE DELIBERATIONS
================================================================================

You can run multiple deliberations concurrently on the same server:

Terminal 1:
    python3 run_deliberation.py --port 2400 --topic "Topic A" \
        --agent LeCun --agent Altman --agent Hinton

Terminal 2:
    python3 run_deliberation.py --port 2400 --topic "Topic B" \
        --agent Bengio --agent Musk --agent Ng

Each deliberation gets a unique ID and tracks its own state.


================================================================================
TROUBLESHOOTING
================================================================================

"Cannot connect to API on port XXXX"
    -> Make sure services are started: ./run.sh 6400 2400 start
    -> Check if port is already in use

"No topic specified"
    -> Add --topic "Your question here" to the command

"Failed to activate personas"
    -> Check if the persona names are valid
    -> Try using --persona-version v2

"0 LLM log entries"
    -> Make sure SWARM_LIVE=true is set when starting services
    -> Check OPENROUTER_API_KEY is set

View server logs for debugging:
    tail -f logs/api_server.log


================================================================================
EXAMPLE: FULL EXPERIMENT RUN
================================================================================

# 1. Set API key (load from file)
export OPENROUTER_API_KEY=$(cat openrouter.txt)

# 2. Start services
SWARM_LIVE=true ./run.sh 6400 2400 start

# 3. Run deliberation
python3 run_deliberation.py --port 2400 \
    --topic "Should AI systems be required to explain their decisions?" \
    --agent "Yann LeCun" \
    --agent "Sam Altman" \
    --agent "Stuart Russell" \
    --agent "Timnit Gebru" \
    --agent "Dario Amodei"

# 4. Watch in browser
#    Open http://localhost:2400

# 5. When done, stop services
./run.sh 6400 2400 stop

================================================================================
