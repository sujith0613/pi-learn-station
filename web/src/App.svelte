<script lang="ts">
  // Word Craft — a kind early-reading writing station.
  //
  // Kids trace/write the target word on a handwriting canvas. Each completed
  // letter strokes get recognized; uncertain letters become tappable tiles;
  // the backend char-LM suggests corrections; the word is spoken aloud.
  import { onMount } from 'svelte'

  const WORDS = ['dog', 'bus', 'bag', 'pig', 'fox', 'sun', 'bed', 'bird', 'box', 'frog']

  let target = WORDS[Math.floor(Math.random() * WORDS.length)]

  // ---- drawing state -------------------------------------------------------
  let canvasEl: HTMLCanvasElement
  let ctx: CanvasRenderingContext2D | null = null
  let drawing = false
  let curStroke: number[][] = []   // [[x,y], ...] in CSS px (not device px)
  let strokes: number[][][] = []   // all strokes of the current word, CSS px

  const PAD_TOP = 0.18    // fraction of canvas height above the baseline band
  const BAND = 0.28       // height of one writing band as fraction
  const STROKE_W = 5

  function resize() {
    if (!canvasEl) return
    const rect = canvasEl.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvasEl.width = Math.round(rect.width * dpr)
    canvasEl.height = Math.round(rect.height * dpr)
    ctx = canvasEl.getContext('2d')
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0)
    redraw()
  }

  function redraw() {
    if (!ctx || !canvasEl) return
    const c = ctx
    const w = canvasEl.width / (window.devicePixelRatio || 1)
    const h = canvasEl.height / (window.devicePixelRatio || 1)
    c.clearRect(0, 0, w, h)

    // handwriting guide lines (3 bands => 4 lines)
    c.strokeStyle = 'rgba(107, 97, 84, 0.28)'
    c.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const y = PAD_TOP * h + i * BAND * h
      c.beginPath()
      c.moveTo(0, y)
      c.lineTo(w, y)
      c.stroke()
    }
    // midline (dashed) to show where lowercase letters' tops stop
    c.strokeStyle = 'rgba(232, 168, 124, 0.4)'
    c.setLineDash([4, 5])
    c.beginPath()
    c.moveTo(0, PAD_TOP * h + BAND * h)
    c.lineTo(w, PAD_TOP * h + BAND * h)
    c.stroke()
    c.setLineDash([])

    // ink
    c.strokeStyle = '#2a2620'
    c.lineWidth = STROKE_W
    c.lineCap = 'round'
    c.lineJoin = 'round'
    for (const stroke of strokes) {
      c.beginPath()
      stroke.forEach(([x, y], i) => (i === 0 ? c.moveTo(x, y) : c.lineTo(x, y)))
      c.stroke()
    }
    if (curStroke.length) {
      c.beginPath()
      curStroke.forEach(([x, y], i) => (i === 0 ? c.moveTo(x, y) : c.lineTo(x, y)))
      c.stroke()
    }
  }

  function pos(ev: PointerEvent): [number, number] {
    const rect = canvasEl.getBoundingClientRect()
    return [ev.clientX - rect.left, ev.clientY - rect.top]
  }

  function down(ev: PointerEvent) {
    if (ev.pointerType !== 'mouse' || ev.isPrimary) drawing = true
    else drawing = true
    canvasEl.setPointerCapture(ev.pointerId)
    curStroke = [pos(ev)]
    redraw()
  }

  function move(ev: PointerEvent) {
    if (!drawing) return
    curStroke.push(pos(ev))
    redraw()
  }

  function up(_ev: PointerEvent) {
    if (!drawing) return
    if (curStroke.length) strokes.push(curStroke)
    curStroke = []
    drawing = false
    redraw()
    check()
  }

  // ---- recognition ---------------------------------------------------------
  let checking = false
  let recognized: { letters: string[][] } | null = null
  let message = ''
  let success = false

  async function check() {
    if (!strokes.length || checking) return
    checking = true
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strokes: [strokes] }),
      })
      const data = await res.json()
      // data.letters: [[[letter, prob], ...], ...]
      const word = (data.letters as string[][][])
        .map((pos) => (pos.length ? pos[0][0] : '?'))
        .join('')
      message = word === target ? 'That looks like a perfect ' + target + '!' : ''
      success = word === target
      recognized = { letters: data.letters.map((p: string[][]) => p.map(([l]) => l as string)) }
      if (success) speak(target)
    } catch {
      message = ''
    } finally {
      checking = false
    }
  }

  // ---- hints & speech ------------------------------------------------------
  let hintLetter: string | null = null

  function speak(text: string) {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 0.9
    u.pitch = 1.05
    u.lang = 'en-US'
    window.speechSynthesis.speak(u)
  }

  function nextWord() {
    target = WORDS[Math.floor(Math.random() * WORDS.length)]
    strokes = []
    recognized = null
    message = ''
    success = false
    hintLetter = null
    redraw()
  }

  function clearCanvas() {
    strokes = []
    recognized = null
    message = ''
    success = false
    hintLetter = null
    redraw()
  }

  onMount(() => {
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvasEl)
    speak('Write ' + target)
    return () => ro.disconnect()
  })
</script>

<svelte:head>
  <meta name="theme-color" content="#FBF6ED" />
</svelte:head>

<main class="desk">
  <header class="topbar">
    <div class="brand">Word&nbsp;Craft</div>
    <div class="actions">
      <button class="btn ghost" onclick={clearCanvas}>Clear</button>
      <button class="btn ghost" onclick={() => speak(target)}>Hear</button>
      <button class="btn solid" onclick={nextWord}>New word</button>
    </div>
  </header>

  <section class="prompt" aria-label="word to write">
    <span class="prompt-label">Write</span>
    <div class="tiles">
      {#each target as ch, i}
        <button
          class="tile {hintLetter === ch ? 'hint' : ''}"
          onclick={() => { hintLetter = ch; speak(ch) }}
          aria-label="letter {ch}"
        >
          {hintLetter === ch ? ch : ''}
        </button>
      {/each}
    </div>
  </section>

  <section class="canvas-wrap" aria-label="writing area">
    <canvas
      bind:this={canvasEl}
      onpointerdown={down}
      onpointermove={move}
      onpointerup={up}
      onpointercancel={up}
    ></canvas>
    {#if checking}
      <div class="busy">reading…</div>
    {/if}
  </section>

  {#if recognized}
    <section class="strip" aria-label="your word">
      {#each recognized.letters as pos, i}
        <button
          class="rcell {pos[0] === target[i] ? 'ok' : 'warn'}"
          onclick={() => speak(pos[0])}
          aria-label="letter {pos[0]}"
        >
          {pos[0]}
          {#if pos.length > 1}<span class="alt">{pos.slice(1, 3).join(' ')}</span>{/if}
        </button>
      {/each}
    </section>
  {/if}

  <section class="feedback" class:success aria-live="polite">
    {#if message}
      <div class="bubble">
        <span class="sticker">{success ? '★' : '·'}</span>
        <span>{message}</span>
        {#if success}
          <button class="btn solid next" onclick={nextWord}>Next word →</button>
        {/if}
      </div>
    {/if}
  </section>

  <footer class="foot">
    {#if strokes.length}<span>{strokes.length} stroke{strokes.length === 1 ? '' : 's'}</span>{/if}
    <button class="btn ghost" onclick={clearCanvas}>erase</button>
  </footer>
</main>

<style>
  .desk {
    height: 100%;
    max-width: 720px;
    margin: 0 auto;
    padding: max(1rem, env(safe-area-inset-top)) 1rem
      calc(1rem + env(safe-area-inset-bottom));
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .brand {
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
  }

  .actions {
    display: flex;
    gap: 0.4rem;
  }

  .btn {
    border: none;
    border-radius: 999px;
    padding: 0.55rem 1rem;
    font-size: 0.95rem;
    font-weight: 700;
    transition: transform 0.08s ease;
  }

  .btn:active {
    transform: scale(0.96);
  }

  .btn.ghost {
    background: var(--paper-deep);
    color: var(--ink);
  }

  .btn.solid {
    background: var(--ink);
    color: var(--paper);
  }

  .btn.next {
    background: var(--sage-deep);
    color: #fff;
  }

  .prompt {
    text-align: center;
  }

  .prompt-label {
    display: block;
    font-size: 0.85rem;
    color: var(--ink-soft);
    margin-bottom: 0.35rem;
  }

  .tiles {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
  }

  .tile {
    width: 2.6rem;
    height: 3rem;
    border-radius: var(--radius-md);
    border: 2px dashed var(--line);
    background: var(--paper-deep);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--ink-soft);
    display: grid;
    place-items: center;
  }

  .tile.hint {
    border-color: var(--coral);
    background: #fff7ec;
    color: var(--ink);
  }

  .canvas-wrap {
    position: relative;
    flex: 1;
    min-height: 240px;
  }

  canvas {
    width: 100%;
    height: 100%;
    background: #fffdf8;
    border: 2px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    touch-action: none;
    display: block;
  }

  .busy {
    position: absolute;
    right: 1rem;
    top: 1rem;
    background: rgba(42, 38, 32, 0.85);
    color: #fff;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    animation: pulse 1s infinite alternate;
  }

  .strip {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
  }

  .rcell {
    min-width: 2.6rem;
    height: 3rem;
    border-radius: var(--radius-md);
    border: 2px solid var(--line);
    background: #fff;
    font-size: 1.5rem;
    font-weight: 800;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  .rcell.ok {
    border-color: var(--sage);
    color: var(--sage-deep);
  }

  .rcell.warn {
    border-color: var(--coral);
    color: var(--ink);
  }

  .alt {
    position: absolute;
    bottom: -0.3rem;
    font-size: 0.55rem;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
  }

  .feedback {
    min-height: 3.4rem;
    display: grid;
    place-items: center;
  }

  .bubble {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: var(--paper-deep);
    border-radius: var(--radius-md);
    padding: 0.7rem 1.1rem;
    font-weight: 600;
  }

  .bubble .sticker {
    font-size: 1.4rem;
  }

  .success .bubble {
    background: var(--sage);
    color: #fff;
  }

  .success .bubble .sticker {
    animation: pop 0.4s ease;
  }

  .foot {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--ink-soft);
    font-size: 0.8rem;
  }

  @keyframes pulse {
    from { opacity: 0.7; }
    to { opacity: 1; }
  }

  @keyframes pop {
    0% { transform: scale(0); }
    70% { transform: scale(1.3); }
    100% { transform: scale(1); }
  }

  @media (max-width: 480px) {
    .brand { font-size: 0.95rem; }
    .tile, .rcell { width: 2.2rem; height: 2.6rem; font-size: 1.25rem; }
  }
</style>