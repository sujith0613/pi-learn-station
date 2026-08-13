<script lang="ts">
  // Free-Write — a kind early-writing correction station.
  //
  // The child writes whatever they choose, freely, on the canvas. The backend
  // auto-segments the strokes into letters and words, then uses a char-level
  // language model to suggest the likely intended spelling of each word.
  // Suggestions are optional, tappable tiles — never blocking.
  import { onMount } from 'svelte'

  type Pt = { x: number; y: number; t: number }
  type Word = {
    start: number
    end: number
    greedy: string
    letters: [string, number][][]
    suggestion?: { best: string; alternatives: string[] }
  }
  type Result = { sentence: string; words: Word[] }

  // ---- drawing state -------------------------------------------------------
  let canvasEl: HTMLCanvasElement
  let ctx: CanvasRenderingContext2D | null = null
  let drawing = false
  let cur: Pt[] = []
  let strokes: Pt[][] = []   // flat list of strokes (each = points with t)

  const PAD_TOP = 0.18
  const BAND = 0.28
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

    c.strokeStyle = 'rgba(107, 97, 84, 0.28)'
    c.lineWidth = 1
    for (let i = 0; i <= 3; i++) {
      const y = PAD_TOP * h + i * BAND * h
      c.beginPath()
      c.moveTo(0, y)
      c.lineTo(w, y)
      c.stroke()
    }
    c.strokeStyle = 'rgba(232, 168, 124, 0.4)'
    c.setLineDash([4, 5])
    c.beginPath()
    c.moveTo(0, PAD_TOP * h + BAND * h)
    c.lineTo(w, PAD_TOP * h + BAND * h)
    c.stroke()
    c.setLineDash([])

    c.strokeStyle = '#2a2620'
    c.lineWidth = STROKE_W
    c.lineCap = 'round'
    c.lineJoin = 'round'
    for (const st of [...strokes, ...(cur.length ? [cur] : [])]) {
      if (!st.length) continue
      c.beginPath()
      st.forEach((p, i) => (i === 0 ? c.moveTo(p.x, p.y) : c.lineTo(p.x, p.y)))
      c.stroke()
    }
  }

  function pos(ev: PointerEvent): Pt {
    const rect = canvasEl.getBoundingClientRect()
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top, t: ev.timeStamp }
  }

  function down(ev: PointerEvent) {
    drawing = true
    canvasEl.setPointerCapture(ev.pointerId)
    cur = [pos(ev)]
    redraw()
  }

  function move(ev: PointerEvent) {
    if (!drawing) return
    cur.push(pos(ev))
    redraw()
  }

  function up(_ev: PointerEvent) {
    if (!drawing) return
    if (cur.length) strokes.push(cur)
    cur = []
    drawing = false
    redraw()
    schedule()
  }

  // ---- recognition (debounced) ---------------------------------------------
  let result: Result | null = null
  let checking = false
  let timer: ReturnType<typeof setTimeout> | null = null

  function schedule() {
    if (timer) clearTimeout(timer)
    timer = setTimeout(check, 500)
  }

  async function check() {
    if (!strokes.length || checking) return
    checking = true
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strokes }),
      })
      const data = await res.json()
      result = data
    } catch {
      /* keep last result */
    } finally {
      checking = false
    }
  }

  // ---- sentence rendering ---------------------------------------------------
  let story = ''   // finalized lines, kept as the child writes more

  function sentenceText(): string {
    if (!result) return ''
    return result.words.map((w) => w.greedy).join(' ')
  }

  function applySuggestion(w: Word) {
    if (!w.suggestion) return
    w.greedy = w.suggestion.best
    w.suggestion = undefined
    result = { ...result!, words: result!.words.map((x) => (x === w ? w : x)) }
    speak(w.greedy)
  }

  function nextLine() {
    const txt = sentenceText()
    if (txt) story = (story ? story + ' ' : '') + txt
    strokes = []
    result = null
    redraw()
  }

  // ---- speech ---------------------------------------------------------------
  function speak(text: string) {
    if (!('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 0.9
    u.pitch = 1.05
    u.lang = 'en-US'
    window.speechSynthesis.speak(u)
  }

  function clearCanvas() {
    strokes = []
    result = null
    redraw()
  }

  function isUncertain(w: Word, ci: number): boolean {
    const pos = w.letters?.[ci]
    if (!pos || !pos.length) return true
    const top = pos[0][1]
    return top < 0.5 || !!w.suggestion
  }

  function hasSuggestions(): boolean {
    return result?.words.some((w) => w.suggestion) ?? false
  }

  onMount(() => {
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvasEl)
    return () => ro.disconnect()
  })
</script>

<svelte:head>
  <meta name="theme-color" content="#FBF6ED" />
</svelte:head>

<main class="desk">
  <header class="topbar">
    <div class="brand">Free&nbsp;Write</div>
    <div class="actions">
      <button class="btn ghost" onclick={clearCanvas}>Clear</button>
      <button class="btn solid" onclick={() => speak(sentenceText() || story)}>Hear</button>
      <button class="btn solid" onclick={nextLine}>New line →</button>
    </div>
  </header>

  {#if story}
    <section class="story" aria-label="what you wrote">
      <span class="story-label">Your writing</span>
      <p class="story-text">{story}</p>
    </section>
  {/if}

  {#if result}
    <section class="sentence" aria-label="your words">
      {#each result.words as w, wi}
        <span class="wordwrap">
          <span class="word">
            {#each w.greedy as ch, ci}
              <span
                role="button"
                tabindex="0"
                class="letter {isUncertain(w, ci) ? 'uncertain' : ''}"
                onclick={() => speak(ch)}
                onkeydown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    speak(ch)
                  }
                }}
                aria-label="letter {ch}"
                >{ch}</span
              >
            {/each}
          </span>
          {#if w.suggestion}
            <button
              class="sugg"
              onclick={() => applySuggestion(w)}
              title="tap to correct"
              aria-label="suggested word {w.suggestion.best}"
            >
              → {w.suggestion.best}
            </button>
          {/if}
          {#if wi < result.words.length - 1}
            <span class="space"></span>
          {/if}
        </span>
      {/each}
    </section>
    <div class="hint" aria-live="polite">
      {#if hasSuggestions()}
        Tap a <span class="chip">→ word</span> chip to fix it.
      {/if}
    </div>
  {/if}

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

  <footer class="foot">
    {#if strokes.length}
      <span>{strokes.length} stroke{strokes.length === 1 ? '' : 's'}</span>
    {:else}
      <span>Write anything you like</span>
    {/if}
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
    cursor: pointer;
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

  .story {
    background: var(--paper-deep);
    border-radius: var(--radius-md);
    padding: 0.7rem 1rem;
  }

  .story-label {
    display: block;
    font-size: 0.8rem;
    color: var(--ink-soft);
    margin-bottom: 0.2rem;
  }

  .story-text {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--sage-deep);
    line-height: 1.6;
    word-spacing: 0.35rem;
  }

  .sentence {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 0.4rem 0.5rem;
    min-height: 3.4rem;
    padding: 0.5rem;
    background: #fffdf8;
    border: 2px solid var(--line);
    border-radius: var(--radius-lg);
  }

  .wordwrap {
    display: inline-flex;
    align-items: flex-start;
    flex-direction: column;
    gap: 0.2rem;
  }

  .word {
    display: inline-flex;
    gap: 0.1rem;
  }

  .letter {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--ink);
    padding: 0.05rem 0.1rem;
    border-radius: 4px;
    cursor: pointer;
  }

  .letter.uncertain {
    color: var(--coral-deep, #c95f3d);
    text-decoration: underline;
    text-decoration-style: dotted;
  }

  .space {
    width: 0.75rem;
  }

  .sugg {
    border: 2px solid var(--sage);
    background: var(--sage);
    color: #fff;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.95rem;
    font-weight: 800;
    cursor: pointer;
    animation: pop 0.25s ease;
    align-self: flex-start;
  }

  .hint {
    text-align: center;
    font-size: 0.85rem;
    color: var(--ink-soft);
    min-height: 1.2rem;
  }

  .hint .chip {
    color: var(--sage-deep);
    font-weight: 800;
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
    0% { transform: scale(0.7); }
    70% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }

  @media (max-width: 480px) {
    .brand { font-size: 0.95rem; }
    .letter { font-size: 1.3rem; }
  }
</style>