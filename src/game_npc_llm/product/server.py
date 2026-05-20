from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from game_npc_llm.product.agent import GameAgent
from game_npc_llm.product.memory import InMemoryMemoryStore
from game_npc_llm.product.models import ChatRequest
from game_npc_llm.product.policy import OpenAICompatiblePolicyClient, RulePolicyClient
from game_npc_llm.product.world import load_world, world_to_dict


def create_app() -> FastAPI:
    app = FastAPI(title="GameNPC-RL", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    world = load_world(os.getenv("NPC_WORLD_PATH") or None)
    base_url = os.getenv("NPC_MODEL_BASE_URL")
    model = os.getenv("NPC_MODEL_NAME", "Qwen3-4B-NPC")
    policy = OpenAICompatiblePolicyClient(base_url, model) if base_url else RulePolicyClient()
    agent = GameAgent(world=world, policy=policy, memory=InMemoryMemoryStore(), states={})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "world": world.id, "policy": type(policy).__name__}

    @app.get("/world")
    def get_world() -> dict:
        return world_to_dict(world)

    @app.get("/demo/cases")
    def demo_cases() -> dict:
        path = Path("data/worlds/kisaragi_harbor/demo_cases.jsonl")
        if not path.exists():
            return {"cases": []}
        cases = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                import json

                cases.append(json.loads(line))
        return {"cases": cases}

    @app.get("/state/{session_id}")
    def get_state(session_id: str) -> dict:
        return agent.state_for(session_id).model_dump(mode="json")

    @app.post("/reset/{session_id}")
    def reset(session_id: str) -> dict:
        return agent.reset(session_id).model_dump(mode="json")

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict:
        try:
            return agent.chat(request.npc_id, request.player_input, request.session_id).model_dump(
                mode="json"
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/memory/search")
    def memory_search(session_id: str, npc_id: str, q: str) -> dict[str, list[str]]:
        return {"hits": agent.memory.search(session_id, npc_id, q)}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return WEB_DEMO_HTML

    return app


app = create_app()


WEB_DEMO_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GameNPC-RL Demo</title>
  <style>
    :root {
      --ink: #17231f;
      --paper: #f5ead2;
      --card: rgba(19, 35, 34, 0.88);
      --line: rgba(226, 182, 107, 0.38);
      --gold: #e2b66b;
      --blue: #7ec4cf;
      --red: #e6785f;
      --muted: #9fb8ad;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--paper);
      font-family: "Alegreya", "Iowan Old Style", Georgia, serif;
      background:
        radial-gradient(circle at 15% 12%, rgba(126,196,207,.28), transparent 28rem),
        radial-gradient(circle at 82% 18%, rgba(226,182,107,.22), transparent 24rem),
        linear-gradient(135deg, #0a1417 0%, #132322 54%, #2d2a20 100%);
      min-height: 100vh;
    }
    main { max-width: 1440px; margin: 0 auto; padding: 28px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 22px; }
    h1 { font-size: clamp(38px, 5vw, 76px); line-height: .9; margin: 0; letter-spacing: -2px; }
    h2 { margin: 0 0 12px; font-size: 22px; }
    p { color: #d5cab4; }
    .badge { border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; color: var(--gold); }
    .grid { display: grid; grid-template-columns: 1.05fr 1.2fr .85fr; gap: 18px; align-items: start; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: 0 24px 80px rgba(0,0,0,.28);
      backdrop-filter: blur(10px);
    }
    .map { display: grid; gap: 10px; }
    .loc {
      border: 1px solid rgba(126,196,207,.35);
      border-radius: 18px;
      padding: 12px;
      background: rgba(7, 17, 18, .5);
    }
    .loc strong { color: var(--blue); }
    .npc-list { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
    .npc {
      text-align: left;
      border: 1px solid var(--line);
      background: rgba(245,234,210,.06);
      color: var(--paper);
      border-radius: 16px;
      padding: 12px;
      cursor: pointer;
    }
    .npc.active { background: rgba(226,182,107,.18); border-color: var(--gold); }
    button, select, input {
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
    }
    button { background: var(--gold); color: var(--ink); font-weight: 800; cursor: pointer; }
    button.secondary { background: transparent; color: var(--paper); }
    input { width: 100%; background: rgba(245,234,210,.95); color: var(--ink); }
    .composer { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; margin-top: 12px; }
    .log { display: grid; gap: 12px; max-height: 58vh; overflow: auto; padding-right: 4px; }
    .turn { border-radius: 18px; padding: 12px; background: rgba(0,0,0,.24); border: 1px solid rgba(255,255,255,.07); }
    .player { border-color: rgba(126,196,207,.32); }
    .npc-turn { border-color: rgba(226,182,107,.32); }
    pre { white-space: pre-wrap; overflow: auto; background: #071112; padding: 14px; border-radius: 16px; color: #dff5ed; }
    .kv { display: grid; gap: 8px; }
    .pill { display: inline-block; margin: 3px; padding: 5px 9px; border-radius: 999px; background: rgba(126,196,207,.14); color: #c7f7ff; }
    .warn { color: var(--red); }
    @media (max-width: 980px) {
      header { display: block; }
      .grid { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      .npc-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Kisaragi Harbor</h1>
      <p>5 NPCs, 3 locations, 2 questlines, structured JSON actions, memory, and quest state.</p>
    </div>
    <div class="badge" id="health">checking service...</div>
  </header>
  <div class="grid">
    <section class="card">
      <h2>Harbor Map</h2>
      <div id="map" class="map"></div>
      <h2 style="margin-top:18px">NPC Cast</h2>
      <div id="npcList" class="npc-list"></div>
    </section>
    <section class="card">
      <h2 id="sceneTitle">Scene</h2>
      <p id="npcInfo"></p>
      <div id="log" class="log"></div>
      <div class="composer">
        <input id="msg" value="The tide engine is overheating. I found the broken valve near the pier." />
        <button onclick="chat()">Talk</button>
        <button class="secondary" onclick="reset()">Reset</button>
      </div>
    </section>
    <aside class="card">
      <h2>Game State</h2>
      <div id="state" class="kv"></div>
      <h2 style="margin-top:18px">Last JSON</h2>
      <pre id="json">Loading world...</pre>
    </aside>
  </div>
</main>
<script>
let world = null;
let selectedNpc = 'mika';
const sessionId = 'web';
const starterLines = [
  'The tide engine is overheating. I found the broken valve near the pier.',
  'Ren, I need proof about the erased docking record.',
  'Hana, the lighthouse warning sounded like a dead captain.',
  'Toma, why did your late ship avoid the main ledger?',
  'IKO-7, route me to someone I can trust.'
];
async function init() {
  const health = await (await fetch('/health')).json();
  document.getElementById('health').textContent = `${health.policy} / ${health.world}`;
  world = await (await fetch('/world')).json();
  renderMap();
  renderNpcs();
  await reset();
}
function renderMap() {
  const map = document.getElementById('map');
  map.innerHTML = '';
  Object.values(world.locations).forEach(loc => {
    const div = document.createElement('div');
    div.className = 'loc';
    div.innerHTML = `<strong>${loc.name}</strong><p>${loc.description}</p>${loc.entities.map(e => `<span class="pill">${e}</span>`).join('')}`;
    map.appendChild(div);
  });
}
function renderNpcs() {
  const list = document.getElementById('npcList');
  list.innerHTML = '';
  Object.values(world.npcs).forEach((n, i) => {
    const button = document.createElement('button');
    button.className = `npc ${n.id === selectedNpc ? 'active' : ''}`;
    button.innerHTML = `<strong>${n.name}</strong><br><small>${n.role}</small>`;
    button.onclick = () => {
      selectedNpc = n.id;
      document.getElementById('msg').value = starterLines[i] || 'What should I do next?';
      renderNpcs();
      showNpc();
    };
    list.appendChild(button);
  });
  showNpc();
}
function showNpc() {
  const n = world.npcs[selectedNpc];
  const loc = world.locations[n.location_id];
  document.getElementById('sceneTitle').textContent = `${n.name} at ${loc.name}`;
  document.getElementById('npcInfo').textContent = `${n.persona} Goals: ${n.goals.join('; ')}`;
}
async function chat() {
  const body = {
    session_id: sessionId,
    npc_id: selectedNpc,
    player_input: document.getElementById('msg').value
  };
  const res = await fetch('/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const payload = await res.json();
  appendTurn('player', body.player_input);
  appendTurn('npc-turn', payload.response.dialogue, payload.response);
  renderState(payload.state, payload.memory_hits, payload.events);
  document.getElementById('json').textContent = JSON.stringify(payload, null, 2);
}
async function reset() {
  const state = await (await fetch(`/reset/${sessionId}`, {method: 'POST'})).json();
  document.getElementById('log').innerHTML = '<div class="turn">Session reset. Choose an NPC and start a quest.</div>';
  renderState(state, [], []);
  document.getElementById('json').textContent = JSON.stringify({world, state}, null, 2);
}
function appendTurn(kind, text, response=null) {
  const div = document.createElement('div');
  div.className = `turn ${kind}`;
  let extra = '';
  if (response) {
    extra = `<div><span class="pill">${response.action}</span>${response.target ? `<span class="pill">${response.target}</span>` : ''}${response.safety_flags.map(f => `<span class="pill warn">${f}</span>`).join('')}</div>`;
  }
  div.innerHTML = `<strong>${kind === 'player' ? 'Player' : world.npcs[selectedNpc].name}</strong><p>${text}</p>${extra}`;
  document.getElementById('log').appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
}
function renderState(state, memories, events) {
  const quest = Object.entries(state.quest_status).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join('');
  const inv = (state.inventory.length ? state.inventory : ['empty']).map(x => `<span class="pill">${x}</span>`).join('');
  document.getElementById('state').innerHTML = `
    <div><strong>Location</strong><br>${world.locations[state.current_location_id].name}</div>
    <div><strong>Inventory</strong><br>${inv}</div>
    <div><strong>Quests</strong><br>${quest}</div>
    <div><strong>Events</strong><br>${(events.length ? events : ['none']).map(x => `<span class="pill">${x}</span>`).join('')}</div>
    <div><strong>Memory Hits</strong><br>${(memories.length ? memories : ['none']).map(x => `<span class="pill">${x}</span>`).join('')}</div>
  `;
}
init();
</script>
</body>
</html>
"""
