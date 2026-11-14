# Planproof frontend

FastAPI serves this interface at `/planwise/` on the same origin as the API.
Start with the [MVP/frontend handoff](../docs/MVP_FRONTEND_HANDOFF.md) for portable
setup, upload, the general Gemini `/ask` endpoint, error states and citation overlays.

The general question panel lets Gemini choose direct vision, algorithms, clean
crops, Nemotron inspection and self-checks. Provider permissions are explicit.
The older door-comparison and synthetic demo controls remain separate flows.

Frontend files: `index.html`, `app.js`, `styles.css`. Keep keys and algorithms on
the server. Render canonical citations over the returned source image and clear
stale answers when the selected page changes. See the [agent contract](../docs/QUESTION_AGENT.md)
and [50-question results](../docs/AGENT_COHORT50.md) for capabilities and limitations.
