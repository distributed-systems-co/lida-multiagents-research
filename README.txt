Setup instructions:
    sudo apt install docker.io docker-compose

    export OPENROUTER_API_KEY=$(cat openrouter.txt)
    docker-compose up --build

Use browser to visit localhost:2040


To see logs, try
    curl "http://localhost:2040/api/llm/logs" | jq | less
Or
    docker container list
    docker cp 56cf246d77ab:/app/data/datasets.db db.db
    sqlite db.db
    sqlite> select prompt,response from llm_response_log limit 10;
Curl shows newest first, sqlite shows oldest first.
