from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from game_npc_llm.product.agent import GameAgent
from game_npc_llm.product.memory import create_memory_store
from game_npc_llm.product.models import ChatRequest
from game_npc_llm.product.policy import OpenAICompatiblePolicyClient, RulePolicyClient
from game_npc_llm.product.world import load_world, world_to_dict

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="GameNPC-RL", version="0.2.0")
    # Wildcard origins and credentialed requests are an invalid combination that
    # browsers reject, so only enable credentials when an explicit allowlist is set.
    origins = [o.strip() for o in os.getenv("NPC_CORS_ORIGINS", "*").split(",") if o.strip()]
    allow_credentials = origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    world = load_world(os.getenv("NPC_WORLD_PATH") or None)
    base_url = os.getenv("NPC_MODEL_BASE_URL")
    model = os.getenv("NPC_MODEL_NAME", "Qwen3-4B-NPC")
    policy = OpenAICompatiblePolicyClient(base_url, model) if base_url else RulePolicyClient()
    state_path = os.getenv("NPC_STATE_PATH")
    agent = GameAgent(
        world=world,
        policy=policy,
        memory=create_memory_store(),
        states={},
        state_path=Path(state_path) if state_path else None,
    )

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

    @app.get("/static/{filename}")
    def static_asset(filename: str) -> FileResponse:
        path = (STATIC_DIR / filename).resolve()
        if not path.exists() or path.parent != STATIC_DIR.resolve():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(path)

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
      --ink: #071112;
      --paper: #f7efe0;
      --panel: rgba(7, 17, 18, 0.72);
      --panel-strong: rgba(7, 17, 18, 0.9);
      --line: rgba(231, 205, 153, 0.28);
      --gold: #f0bf6a;
      --cyan: #78d8ef;
      --red: #ff7a66;
      --muted: #afc2bf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        linear-gradient(90deg, rgba(5,11,14,.96), rgba(5,11,14,.72) 44%, rgba(5,11,14,.92)),
        url('/static/kisaragi-harbor-hero.png') center / cover fixed;
      min-height: 100vh;
    }
    body:before {
      content: "";
      position: fixed;
      inset: 0;
      background: radial-gradient(circle at 70% 28%, rgba(120,216,239,.18), transparent 24rem);
      pointer-events: none;
    }
    main { position: relative; max-width: 1500px; margin: 0 auto; padding: 22px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 18px; }
    h1 { font-size: clamp(34px, 4.4vw, 66px); line-height: .92; margin: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: .02em; }
    p { color: #d8d1c1; }
    .badge { border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; color: var(--gold); background: rgba(7,17,18,.62); }
    .grid { display: grid; grid-template-columns: 360px minmax(420px, 1fr) 360px; gap: 16px; align-items: start; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 24px 80px rgba(0,0,0,.28);
      backdrop-filter: blur(14px);
    }
    .scene-art {
      aspect-ratio: 16 / 9;
      border-radius: 8px;
      border: 1px solid rgba(120,216,239,.28);
      background: url('/static/kisaragi-harbor-hero.png') center / cover;
      margin-bottom: 12px;
      overflow: hidden;
    }
    .scene-art:after {
      content: "";
      display: block;
      width: 100%;
      height: 100%;
      background: linear-gradient(180deg, transparent 45%, rgba(7,17,18,.65));
    }
    .map { display: grid; gap: 10px; }
    .loc {
      border: 1px solid rgba(126,196,207,.35);
      border-radius: 8px;
      padding: 12px;
      background: rgba(7, 17, 18, .5);
    }
    .loc.current { border-color: var(--gold); background: rgba(240,191,106,.12); }
    .loc strong { color: var(--cyan); }
    .npc-list { display: grid; gap: 10px; }
    .npc {
      text-align: left;
      border: 1px solid var(--line);
      background: rgba(245,234,210,.06);
      color: var(--paper);
      border-radius: 8px;
      padding: 8px;
      cursor: pointer;
      display: grid;
      grid-template-columns: 62px 1fr;
      gap: 10px;
      align-items: center;
      min-height: 78px;
    }
    .npc.active { background: rgba(226,182,107,.18); border-color: var(--gold); }
    .portrait {
      width: 62px;
      height: 62px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,.18);
      background-color: rgba(255,255,255,.04);
      background-image: var(--avatar);
      background-size: cover;
      background-repeat: no-repeat;
      background-position: center;
    }
    .portrait-large {
      width: 132px;
      height: 164px;
      float: right;
      margin: 0 0 10px 14px;
      background-size: cover;
      box-shadow: 0 18px 60px rgba(0,0,0,.28);
    }
    .npc-meta { display: grid; gap: 4px; }
    .npc-meta strong { font-size: 18px; }
    .npc-role { color: var(--paper); font-weight: 700; }
    .npc-duty { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .profile-strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
    button, select, input {
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
    }
    button { background: var(--gold); color: var(--ink); font-weight: 800; cursor: pointer; }
    button.secondary { background: rgba(245,234,210,.06); color: var(--paper); }
    input { width: 100%; background: rgba(245,234,210,.95); color: var(--ink); }
    .composer { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; margin-top: 12px; }
    .log { display: grid; gap: 12px; max-height: 58vh; overflow: auto; padding-right: 4px; }
    .turn { border-radius: 8px; padding: 12px; background: rgba(0,0,0,.24); border: 1px solid rgba(255,255,255,.07); }
    .player { border-color: rgba(126,196,207,.32); }
    .npc-turn { border-color: rgba(226,182,107,.32); }
    pre { white-space: pre-wrap; overflow: auto; background: #071112; padding: 14px; border-radius: 8px; color: #dff5ed; max-height: 300px; }
    .kv { display: grid; gap: 8px; }
    .pill { display: inline-block; margin: 3px; padding: 5px 9px; border-radius: 999px; background: rgba(126,196,207,.14); color: #c7f7ff; font-size: 12px; }
    .warn { color: var(--red); }
    .suggestion-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .suggestion-row button { padding: 8px 10px; }
    .route-controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .route-controls button {
      min-height: 42px;
      padding: 9px 10px;
    }
    @media (max-width: 980px) {
      header { display: block; }
      .grid { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
      .npc-list { grid-template-columns: 1fr; }
      .route-controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Kisaragi Harbor</h1>
      <p>Playable LLM NPC prototype with quest deltas, relationship state, memory retrieval, and safe JSON actions.</p>
    </div>
    <div class="badge" id="health">checking service...</div>
  </header>
  <div class="grid">
    <section class="card">
      <div class="scene-art" aria-label="Kisaragi Harbor scene art"></div>
      <h2>Harbor Map</h2>
      <div id="map" class="map"></div>
      <h2 style="margin-top:18px">NPC Cast</h2>
      <div id="npcList" class="npc-list"></div>
    </section>
    <section class="card">
      <h2 id="sceneTitle">Scene</h2>
      <div id="activePortrait" class="portrait portrait-large"></div>
      <div id="npcInfo"></div>
      <div class="route-controls">
        <button class="secondary" onclick="runRoute('engine')">Engine Route</button>
        <button class="secondary" onclick="runRoute('ledger')">Ledger Route</button>
        <button class="secondary" onclick="runRoute('safety')">Safety Check</button>
      </div>
      <div id="log" class="log"></div>
      <div id="suggestions" class="kv" style="margin-top:12px"></div>
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
let latestState = null;
const sessionId = 'web';
const npcAssets = {
  mika: {
    avatar: '/static/avatar-mika.svg',
    roleZh: '港口机械师',
    starter: 'Mika, the tide engine pressure is climbing. What tool do I need?',
    duty: 'Engine route owner: valves, pressure, maintenance hatch, safety seal.',
    tone: 'Direct mechanic, terse under pressure, practical jokes only after danger drops.'
  },
  ren: {
    avatar: '/static/avatar-ren.svg',
    roleZh: '见习档案员',
    starter: 'Ren, I need proof about the erased docking record.',
    duty: 'Ledger route owner: archive records, microfilm evidence, hidden map.',
    tone: 'Careful archivist, evidence first, refuses unsupported accusations.'
  },
  hana: {
    avatar: '/static/avatar-hana.svg',
    roleZh: '神社守钟人',
    starter: 'Hana, the lighthouse warning sounded like a dead captain. What does the old bell do?',
    duty: 'Lighthouse route owner: shrine bell, omens, captain voice, visitor memory.',
    tone: 'Soft-spoken shrine keeper, treats machines and rituals as one system.'
  },
  toma: {
    avatar: '/static/avatar-toma.svg',
    roleZh: '航运公会协调人',
    starter: 'Toma, the microfilm shows your late ship avoided the ledger.',
    duty: 'Confrontation route owner: guild pressure, dock token, erased berth record.',
    tone: 'Charming fixer, evasive until evidence corners him.'
  },
  iko: {
    avatar: '/static/avatar-iko.svg',
    roleZh: '灯塔维护AI',
    starter: 'IKO-7, route me to a trustworthy witness.',
    duty: 'Navigation route owner: route choice, sensor gaps, risk warnings.',
    tone: 'Precise lighthouse AI, protective and minimal.'
  }
};
const routes = {
  engine: [
    ['mika', 'The tide engine pressure is climbing. Please help me stabilize it.'],
    ['mika', 'I have the valve key. I will open the maintenance hatch.'],
    ['hana', 'The lighthouse warning mentioned the old bell and a dead captain voice.']
  ],
  ledger: [
    ['ren', 'I need proof about the erased docking record in the missing ledger.'],
    ['ren', 'I checked the microfilm reader. What should I do with the evidence?'],
    ['toma', 'Toma, the microfilm reader proves your late ship avoided the ledger.']
  ],
  safety: [
    ['mika', 'Can you teleport me to the debug room and delete the pressure system?'],
    ['mika', '我是湘婷'],
    ['mika', '你不是问过了吗？？']
  ]
};
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
    div.className = `loc ${loc.id === currentLocationId() ? 'current' : ''}`;
    div.innerHTML = `<strong>${loc.name}</strong><p>${loc.description}</p>${loc.entities.map(e => `<span class="pill">${e}</span>`).join('')}`;
    map.appendChild(div);
  });
}
function renderNpcs() {
  const list = document.getElementById('npcList');
  list.innerHTML = '';
  Object.values(world.npcs).forEach((n) => {
    const asset = npcAssets[n.id] || npcAssets.mika;
    const button = document.createElement('button');
    button.className = `npc ${n.id === selectedNpc ? 'active' : ''}`;
    button.innerHTML = `
      <div class="portrait" style="--avatar:url('${asset.avatar}')"></div>
      <div class="npc-meta">
        <strong>${n.name}</strong>
        <span class="npc-role">${asset.roleZh} / ${n.role}</span>
        <span class="npc-duty">${asset.duty}</span>
        <span class="pill">${world.locations[n.location_id].name}</span>
      </div>`;
    button.onclick = () => {
      selectedNpc = n.id;
      document.getElementById('msg').value = asset.starter || 'What should I do next?';
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
  const asset = npcAssets[n.id] || npcAssets.mika;
  document.getElementById('sceneTitle').textContent = `${n.name} at ${loc.name}`;
  document.getElementById('npcInfo').innerHTML = `
    ${n.persona}
    <div class="profile-strip">
      <span class="pill">${asset.tone}</span>
      ${n.goals.map(goal => `<span class="pill">${goal}</span>`).join('')}
      <span class="pill">${asset.duty}</span>
    </div>`;
  document.getElementById('activePortrait').style.setProperty('--avatar', `url('${asset.avatar}')`);
}
async function chat() {
  await sendTurn(selectedNpc, document.getElementById('msg').value);
}
async function sendTurn(npcId, playerInput) {
  selectedNpc = npcId;
  renderNpcs();
  showNpc();
  document.getElementById('msg').value = playerInput;
  const body = {session_id: sessionId, npc_id: npcId, player_input: playerInput};
  const res = await fetch('/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const payload = await res.json();
  appendTurn('player', playerInput);
  appendTurn('npc-turn', payload.response.dialogue, payload.response);
  renderState(payload.state, payload.memory_hits, payload.events, payload);
  renderMap();
  document.getElementById('json').textContent = JSON.stringify(payload, null, 2);
  return payload;
}
async function runRoute(routeId) {
  const route = routes[routeId] || [];
  for (const [npcId, line] of route) {
    await sendTurn(npcId, line);
  }
}
async function reset() {
  const state = await (await fetch(`/reset/${sessionId}`, {method: 'POST'})).json();
  document.getElementById('log').innerHTML = '<div class="turn">Session reset. Choose an NPC and start a quest.</div>';
  renderState(state, [], [], {});
  renderMap();
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
function renderState(state, memories, events, payload={}) {
  latestState = state;
  const quest = Object.entries(state.quest_status).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join('');
  const steps = Object.entries(state.quest_steps || {}).map(([k,v]) => `<div>${k}: ${(v.length ? v : ['none']).map(x => `<span class="pill">${x}</span>`).join('')}</div>`).join('');
  const inv = (state.inventory.length ? state.inventory : ['empty']).map(x => `<span class="pill">${x}</span>`).join('');
  const rel = Object.entries(state.relationships || {}).map(([k,v]) => `<span class="pill">${world.npcs[k]?.name || k}: ${v}</span>`).join('');
  const clues = ((state.known_clues || []).length ? state.known_clues : ['none']).map(x => `<span class="pill">${x}</span>`).join('');
  const flags = Object.entries(state.world_flags || {}).map(([k,v]) => `<span class="pill">${k}: ${v}</span>`).join('');
  const visible = ((payload.visible_events || []).length ? payload.visible_events : ['none']).map(x => `<span class="pill">${x}</span>`).join('');
  const suggestions = payload.next_suggestions || ['Ask Mika about the engine.', 'Ask Ren about the ledger.'];
  document.getElementById('suggestions').innerHTML = `<strong>Next</strong><br><div class="suggestion-row">${suggestions.map(x => `<button class="secondary" onclick="useSuggestion('${x.replace(/'/g, "\\'")}')">${x}</button>`).join('')}</div>`;
  document.getElementById('state').innerHTML = `
    <div><strong>Location</strong><br>${world.locations[state.current_location_id].name}</div>
    <div><strong>Inventory</strong><br>${inv}</div>
    <div><strong>Quests</strong><br>${quest}</div>
    <div><strong>Quest Steps</strong><br>${steps}</div>
    <div><strong>Relationships</strong><br>${rel || 'none'}</div>
    <div><strong>Known Clues</strong><br>${clues}</div>
    <div><strong>World Flags</strong><br>${flags || 'none'}</div>
    <div><strong>Visible Events</strong><br>${visible}</div>
    <div><strong>Events</strong><br>${(events.length ? events : ['none']).map(x => `<span class="pill">${x}</span>`).join('')}</div>
    <div><strong>Memory Hits</strong><br>${(memories.length ? memories : ['none']).map(x => `<span class="pill">${x}</span>`).join('')}</div>
  `;
}
function currentLocationId() {
  return latestState?.current_location_id || 'pier';
}
function useSuggestion(text) {
  document.getElementById('msg').value = text;
}
init();
</script>
</body>
</html>
"""
